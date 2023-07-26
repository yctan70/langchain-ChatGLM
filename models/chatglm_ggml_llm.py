from abc import ABC
from langchain.chains.base import Chain
from typing import Any, Dict, List, Optional, Generator
from langchain.callbacks.manager import CallbackManagerForChainRun
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList
from models.loader import LoaderCheckPoint
from models.base import (BaseAnswer,
                         AnswerResult,
                         AnswerResultStream,
                         AnswerResultQueueSentinelTokenListenerQueue)
import torch
import transformers


class ChatGLMGGMLLLMChain(BaseAnswer, Chain, ABC):
    max_token: int = 2048
    temperature: float = 0.95
    # 相关度
    top_p = 0.7
    # 候选词数量
    top_k = 0
    checkPoint: LoaderCheckPoint = None
    # history = []
    history_len: int = 10
    streaming_key: str = "streaming"  #: :meta private:
    history_key: str = "history"  #: :meta private:
    prompt_key: str = "prompt"  #: :meta private:
    output_key: str = "answer_result_stream"  #: :meta private:

    def __init__(self, checkPoint: LoaderCheckPoint = None):
        super().__init__()
        self.checkPoint = checkPoint

    @property
    def _chain_type(self) -> str:
        return "ChatGLMGGMLLLMChain"

    @property
    def _check_point(self) -> LoaderCheckPoint:
        return self.checkPoint

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return [self.prompt_key]

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_key]

    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Generator]:
        generator = self.generatorAnswer(inputs=inputs, run_manager=run_manager)
        return {self.output_key: generator}

    def _generate_answer(self,
                         inputs: Dict[str, Any],
                         run_manager: Optional[CallbackManagerForChainRun] = None,
                         generate_with_callback: AnswerResultStream = None) -> None:
        history = inputs[self.history_key]
        streaming = inputs[self.streaming_key]
        prompt = inputs[self.prompt_key]
        print(f"__call:{prompt}")

        flat_list = [item for sublist in history for item in sublist]
        flat_list.append(prompt)

        if streaming:
            output = ""
            for piece in self.checkPoint.model.stream_chat(
                    flat_list,
                    max_length=self.max_token,
                    do_sample=self.temperature > 0,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
            ):
                output += piece
            self.checkPoint.clear_torch_cache()
            flat_list.append(output)
            history = [[flat_list[i], flat_list[i+1]] for i in range(0, len(flat_list), 2)]
            answer_result = AnswerResult()
            answer_result.history = history
            answer_result.llm_output = {"answer": output}
            generate_with_callback(answer_result)

        else:
            response, _ = self.checkPoint.model.chat(
                flat_list,
                max_length=self.max_token,
                do_sample=self.temperature > 0,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            )
            self.checkPoint.clear_torch_cache()
            flat_list.append(response)
            history = [[flat_list[i], flat_list[i+1]] for i in range(0, len(flat_list), 2)]
            answer_result = AnswerResult()
            answer_result.history = history
            answer_result.llm_output = {"answer": response}

            generate_with_callback(answer_result)

