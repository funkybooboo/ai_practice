import conversationRepository from '../repositories/conversation.repository';
import template from '../prompts/dwight_chatbot/template.txt';
import dwight_schrute from '../prompts/dwight_chatbot/quotes.txt';
import llmClient from '../llm/client';

const instructions = template.replace('{{quotes}}', dwight_schrute);

export type ChatResponse = {
    id: string;
    message: string;
};

export default {
    async sendMessage(
        prompt: string,
        conversationId: string
    ): Promise<ChatResponse> {
        const { text: message, id } = await llmClient.generateText({
            model: 'gpt-4o-mini',
            instructions,
            prompt,
            temperature: 0.2,
            maxTokens: 100,
            previousResponseId:
                conversationRepository.getLastResponseId(conversationId),
        });

        conversationRepository.setLastResponseId(conversationId, id);

        return {
            id: id,
            message,
        };
    },
};
