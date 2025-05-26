### Generate Text in Streaming
import asyncio

from tensorrt_llm import LLM, SamplingParams


def main():

    # model could accept HF model name or a path to local HF model.
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Create a sampling params.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # Async based on Python coroutines
    async def task(id: int, prompt: str):

        # streaming=True is used to enable streaming generation.
        async for output in llm.generate_async(prompt,
                                               sampling_params,
                                               streaming=True):
            print(f"Generation for prompt-{id}: {output.outputs[0].text!r}")

    async def main():
        tasks = [task(id, prompt) for id, prompt in enumerate(prompts)]
        await asyncio.gather(*tasks)

    asyncio.run(main())


if __name__ == '__main__':
    main()