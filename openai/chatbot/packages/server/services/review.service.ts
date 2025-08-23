import type { Review } from "../generated/prisma";
import reviewRepository from "../repositories/review.repository";

export default {
    async getReviews(productId: number): Promise<Review[]> {
        return reviewRepository.getReviews(productId);
    }
};
