import type { Request, Response } from 'express';
import reviewService from '../services/review.service';
import productService from '../services/product.service';
import reviewRepository from '../repositories/review.repository';

export default {
    async getReviews(req: Request, res: Response) {
        const productId = Number(req.params.id);
        if (isNaN(productId)) {
            res.status(400).json({ error: 'Invalid product ID.' });
            return;
        }

        const reviews = await reviewService.getReviews(productId);

        res.json(reviews);
    },

    async summerizeReviews(req: Request, res: Response) {
        const productId = Number(req.params.id);
        if (isNaN(productId)) {
            res.status(400).json({ error: 'Invalid product ID.' });
            return;
        }

        const product = await productService.getProduct(productId);
        if (!product) {
            res.status(400).json({ error: 'Invalid product.' });
            return;
        }

        const reviews = await reviewRepository.getReviews(productId, 1);
        if (!reviews.length) {
            res.status(400).json({ error: 'There are no reviews to summarize.' });
            return;
        }

        const summary = await reviewService.summarizeReviews(productId);

        res.json({ summary });
    },
};
