import type { Review } from "../generated/prisma";
import { PrismaClient } from '../generated/prisma';

export default {
    async getReviews(productId: number): Promise<Review[]> {
        const prisma = new PrismaClient();
        return await prisma.review.findMany({
            where: { productId },
            orderBy: { createdAt: 'desc' }
        });
    }
};
