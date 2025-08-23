import axios from 'axios';
import StarRating, { type Rating } from './StarRating';
import Skeleton from 'react-loading-skeleton';
import { useQuery } from '@tanstack/react-query';
import { Button } from '../ui/button';
import { HiSparkles } from "react-icons/hi";
import { useState } from 'react';

type Props = {
    productId: number;
};

type Review = {
    id: number;
    author: string;
    content: string;
    rating: Rating;
    createdAt: string;
};

type GetReviewsResponse = {
    summary: string | null;
    reviews: Review[];
};

type GetSummaryResponse = {
    summary: string;
};

const ReviewList = ({ productId }: Props) => {
    const [summary, setSummary] = useState('');

    const { data: reviewData, isLoading, error } = useQuery<GetReviewsResponse>({
        queryKey: ['reviews', productId],
        queryFn: async () => {
            const { data } = await axios.get<GetReviewsResponse>(
                `/api/products/${productId}/reviews`
            );
            return data;
        },
    });

    const handleSummarize = async () => {
        const { data } = await axios.post<GetSummaryResponse>(`/api/products/${productId}/reviews/summarize`);
        setSummary(data.summary);
    };

    if (isLoading) {
        return (
            <div className='flex flex-col gap-5'>
                {[1, 2, 3].map(i => (
                    <div key={i}>
                        <Skeleton width={150} />
                        <Skeleton width={100} />
                        <Skeleton count={2} />
                    </div>
                ))}
            </div>
        );
    }

    if (error) {
        return <p className='text-red-500'>Could not get reviews. Try again!</p>
    }

    if (!reviewData?.reviews.length) {
        return null;
    }

    const currentSummary: string = reviewData.summary || summary;

    return (
        <div>
            <div className='mb-5'>
                {currentSummary ? (
                    <p>{currentSummary}</p>
                ) : (
                    <Button onClick={handleSummarize}><HiSparkles />Summarize</Button>
                )}
            </div>

            <div className="flex flex-col gap-5">
                {reviewData?.reviews.map((review) => (
                    <div key={review.id}>
                        <div className="font-semibold">{review.author}</div>
                        <div><StarRating value={review.rating}/></div>
                        <p className="py-2">{review.content}</p>
                    </div>
                ))}
            </div>
        </div>
        
    );
};

export default ReviewList;
