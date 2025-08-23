import OpenAI from 'openai';
import { InferenceClient } from "@huggingface/inference";

const defaultModel = process.env.CHATBOT_MODEL;

const inferenceClient = new InferenceClient(process.env.CHATBOT_HF_TOKEN);
const openaiClient = new OpenAI({
    apiKey: process.env.CHATBOT_OPENAI_API_KEY,
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

type GenerateTextFunction = (options: GenerateTextOptions) => Promise<GenerateTextResponse>;

const openaiGenerateText: GenerateTextFunction = async (options: GenerateTextOptions) => {
    try {
        const response = await openaiClient.responses.create({
            model: options.model,
            instructions: options.instructions,
            input: options.prompt,
            temperature: options.temperature,
            max_output_tokens: options.maxTokens,
            previous_response_id: options.previousResponseId,
        });

        return { text: response.output_text, id: response.id };
    } catch (error) {
        console.error("Error generating text with OpenAI:", error);
        throw error;
    }
};

const metaLlamaGenerateText: GenerateTextFunction = async (options: GenerateTextOptions) => {
    try {
        const chatCompletion = await inferenceClient.chatCompletion({
            provider: "fireworks-ai",
            model: "meta-llama/Llama-3.1-8B-Instruct",
            messages: [
                {
                    role: 'system',
                    content: options.instructions,
                },
                {
                    role: "user",
                    content: options.prompt,
                },
            ],
            max_tokens: options.maxTokens,
            temperature: options.temperature,
            previous_response_id: options.previousResponseId,
        });

        if (chatCompletion.choices.length === 0) {
            return { text: '', id: '' };
        } else {
            const choice = chatCompletion.choices[0];
            const content = choice?.message?.content ?? '';
            const id = chatCompletion.id ?? '';

            return { text: content, id: id };
        }
    } catch (error) {
        console.error("Error generating text with Meta Llama:", error);
        throw error;
    }
};

export default {
    async generateText(options: GenerateTextOptions): Promise<GenerateTextResponse> {
        if(!options.model) {
            options.model = defaultModel;
        }

        if (!options.temperature) {
            options.temperature = 0.2;
        }

        if (!options.maxTokens) {
            options.maxTokens = 300;
        }

        const modifiedPrompt = `${options.prompt}\n\nYou have a max of ${options.maxTokens} tokens to respond with.`;
        options.prompt = modifiedPrompt;

        if (options.model === 'meta-llama/Llama-3.1-8B-Instruct') {
            return metaLlamaGenerateText(options);
        } else {
            return openaiGenerateText(options);
        }
    },
};
