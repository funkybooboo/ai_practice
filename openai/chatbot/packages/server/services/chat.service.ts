import OpenAI from 'openai';

import conversationRepository from '../repositories/conversation.repository';
import template from '../prompts/chatbot.txt';
import dwight_schrute from '../prompts/dwight_schrute.txt';

const instructions = template.replace('{{quotes}}', dwight_schrute);

const client = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});

export type ChatResponse = {
    id: string;
    message: string;
};

export default {
    async sendMessage(
        prompt: string,
        conversationId: string
    ): Promise<ChatResponse> {
        const reponse = await client.responses.create({
            model: 'gpt-4o-mini',
            instructions,
            input: prompt,
            temperature: 0.2,
            max_output_tokens: 100,
            previous_response_id:
                conversationRepository.getLastResponseId(conversationId),
        });

        conversationRepository.setLastResponseId(conversationId, reponse.id);

        return {
            id: reponse.id,
            message: reponse.output_text,
        };
    },
};
