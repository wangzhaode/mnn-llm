import cmnnllm as __cmnnllm

class LLM(__cmnnllm.LLM):
    def load(self, model_dir: str):
        '''
        load model from model_dir

        Parameters
        ----------
        model_dir : model path (split) or model name (single)

        Returns
        -------
        None

        Example:
        -------
        >>> llm.load('../qwen-1.8b-int4')
        '''
        super.load(model_dir)

    def generate(self, input_ids: list):
        '''
        generate by input_ids

        Parameters
        ----------
        input_ids : input token ids, list of int

        Returns
        -------
        output_ids : output token ids, list of int

        Example:
        -------
        >>> input_ids = [151644, 872, 198, 108386, 151645, 198, 151644, 77091]
        >>> output_ids = qwen.generate(input_ids)
        '''
        return super.generate(input_ids)

    def response(self, prompt: str, stream: bool = False):
        '''
        response by prompt

        Parameters
        ----------
        prompt : input prompt
        stream : generate string stream, default is False

        Returns
        -------
        res : output string

        Example:
        -------
        >>> res = qwen.response('Hello', True)
        '''
        return super.response(prompt, stream)

def create(model_dir: str, model_type: str = 'auto'):
    '''
    create LLM instance, type decide by `model_dir` or `model_type`

    Parameters
    ----------
    model_dir : model path or model name contain model type
    model_type : model type, defult is `auto`

    Returns
    -------
    llm : LLM instance

    Example:
    -------
    >>> qwen = mnnllm.create('../qwen-1.8b-int4.mnn')
    '''
    return __cmnnllm.create(model_dir, model_type)