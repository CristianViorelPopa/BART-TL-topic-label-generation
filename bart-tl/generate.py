import argparse
import copy
from typing import List, Tuple

import torch
from fairseq.models.bart import BARTModel
from fairseq.models.bart.hub_interface import BARTHubInterface


class MultiSampleBARTModel(BARTModel):
    """
    The classic BARTModel class does not support sampling multiple results from the beam-search.
    This extension fixes that, returning a number of samples passed by the user.
    """
    @classmethod
    def from_pretrained(
            cls,
            model_name_or_path,
            checkpoint_file='model.pt',
            data_name_or_path='.',
            bpe='gpt2',
            **kwargs,
    ):
        from fairseq import hub_utils
        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )
        return MultiSampleBARTHubInterface(x['args'], x['task'], x['models'][0])


class MultiSampleBARTHubInterface(BARTHubInterface):
    """
    The classic BARTHubInterface class does not support sampling multiple results from the
    beam-search. This extension fixes that, returning a number of samples passed by the user.
    """
    def sample(self, sentences: List[str], beam: int = 1, verbose: bool = False,
               num_samples: int = 1, **kwargs) -> List[List]:
        assert num_samples <= beam
        input = [self.encode(sentence) for sentence in sentences]
        sample, translations = self.generate(input, beam, verbose, **kwargs)
        results = []

        for translation in translations:
            results.append([])
            for idx in range(num_samples):
                results[-1].append([self.decode(x['tokens'])
                                    for x in [v for _, v in sorted(zip(sample['id'].tolist(),
                                                                       [translation[idx]]))]][0])
        return results

    def generate(self, tokens: List[torch.LongTensor], beam: int = 5, verbose: bool = False,
                 **kwargs) -> Tuple:
        sample = self._build_sample(tokens)

        # build generator using current args as well as any kwargs
        gen_args = copy.copy(self.args)
        gen_args.beam = beam
        for k, v in kwargs.items():
            setattr(gen_args, k, v)
        generator = self.task.build_generator(gen_args)
        translations = self.task.inference_step(
            generator,
            [self.model],
            sample,
            prefix_tokens=sample['net_input']['src_tokens'].new_zeros((len(tokens), 1)).fill_(self.task.source_dictionary.bos()),
        )

        if verbose:
            src_str_with_unk = self.string(tokens)
            print('S\t{}'.format(src_str_with_unk))

        # return the sample and raw translations and process them in the `sample` method
        return sample, translations

    def forward(self, x):
        return self.sample(x)


def main():
    parser = argparse.ArgumentParser(
        description='Script for generating the labels for topics using a BART model'
    )
    parser.add_argument('--model-path', '-m', required=True, type=str,
                        help='Path to the BART model (trained using PyTorch)')
    parser.add_argument('--processed-dataset-path', '-d', required=True, type=str,
                        help='Path to the processed dataset that the BART model was trained on')
    parser.add_argument('--topics-file', '-i', required=True, type=str,
                        help='The file that contains the topics you want to generate labels for')
    parser.add_argument('--output-file', '-o', required=True, type=str,
                        help='Where the generated labels will be saved to')

    args = parser.parse_args()

    bart = MultiSampleBARTModel.from_pretrained(
        '.',
        checkpoint_file=args.model_path,
        data_name_or_path=args.processed_dataset_path
    )

    # bart.cuda()
    bart.eval()
    # bart.half()
    count = 1
    bsz = 32
    beam = 25
    num_samples = 10

    with open(args.topics_file) as source, open(args.output_file, 'w') as fout:
        sline = source.readline().strip()
        slines = [sline]
        for sline in source:
            if count % bsz == 0:
                with torch.no_grad():
                    hypotheses_batch = []
                    for topic in slines:
                        hypotheses_batch.append(bart.sample([topic], num_samples=num_samples, beam=beam, lenpen=2.0, max_len_b=60, min_len=1, no_repeat_ngram_size=10, length_penalty=1.0)[0])

                for hypotheses in hypotheses_batch:
                    fout.write(' '.join([hypo.replace(' ', '_') for hypo in hypotheses]) + '\n')
                    fout.flush()
                slines = []
                print('Done one batch.')

            slines.append(sline.strip())
            count += 1
        if slines != []:
            hypotheses_batch = []
            for topic in slines:
                hypotheses_batch.append(bart.sample([topic], num_samples=num_samples, beam=beam, lenpen=2.0, max_len_b=60, min_len=1, no_repeat_ngram_size=10, length_penalty=1.0)[0])
            for hypotheses in hypotheses_batch:
                fout.write(' '.join([hypo.replace(' ', '_') for hypo in hypotheses]) + '\n')
                fout.flush()


if __name__ == '__main__':
    main()
