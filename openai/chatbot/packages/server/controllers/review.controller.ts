import type { Request, Response } from 'express';
import reviewService from '../services/review.service';

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

        const summary = await reviewService.summarizeReviews(productId);

        // TODO save summary to db

        res.json({ summary });
    },
};
