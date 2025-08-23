import OpenAI from "openai";

const client = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});

type GenerateTextOptions = {
    model?: string;
    instructions?: string;
    prompt: string;
    temperature?: number;
    maxTokens?: number;
    previousResponseId?: string;
};

type GenerateTextResponse = { 
    text: string;
    id: string;
};

export default {
    async generateText({ model = 'gpt-4.1', instructions, prompt, temperature = 0.2, maxTokens = 300, previousResponseId }: GenerateTextOptions): Promise<GenerateTextResponse> {
        const modIfiedPrompt = prompt + `\n\n You have a max of ${maxTokens} tokens to responsed with.`

        const repsonse = await client.responses.create({
            model,
            instructions,
            input: modIfiedPrompt,
            temperature,
            max_output_tokens: maxTokens,
            previous_response_id: previousResponseId
        });

        return { text: repsonse.output_text, id: repsonse.id};
    }
};
