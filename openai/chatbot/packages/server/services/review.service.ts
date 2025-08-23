import type { Review } from '../generated/prisma';
import reviewRepository from '../repositories/review.repository';
import llmClient from '../llm/client';
import template from '../prompts/summerize_reviews/template.txt';

export default {
    async getReviews(productId: number): Promise<Review[]> {
        return reviewRepository.getReviews(productId);
    },

    async summarizeReviews(productId: number): Promise<string> {
        const existingSummary = await reviewRepository.getReviewSummary(productId);
        if (existingSummary && existingSummary.expiresAt > new Date()) {
            return existingSummary.content;
        }

        const reviews = await reviewRepository.getReviews(productId, 10);
        const joinedReivews = reviews.map((r) => r.content).join('\n\n');

        const prompt = template.replace('{{reivews}}', joinedReivews);

        const { text: summary } = await llmClient.generateText({
            prompt,
            maxTokens: 500,
        });

        await reviewRepository.storeReviewSummary(productId, summary);

        return summary;
    },
};
