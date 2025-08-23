import axios from 'axios';
import StarRating, { type Rating } from './StarRating';
import { useQuery } from '@tanstack/react-query';
import { Button } from '../ui/button';
import { HiSparkles } from "react-icons/hi";
import { useState } from 'react';
import ReviewSkeleton from './ReviewSkeleton';

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
    const [isSummaryLoading, setIsSummaryLoading] = useState(false);
    const [summaryError, setSummaryError] = useState('');

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
        try {
            setIsSummaryLoading(true);
            setSummaryError('');

            const { data } = await axios.post<GetSummaryResponse>(`/api/products/${productId}/reviews/summarize`);

            setSummary(data.summary);
        } catch (error) {
            console.error(error);
            setSummaryError('Could not summarize the reviews. Try Again!');
        } finally {
            setIsSummaryLoading(false);
        }
    };

    if (isLoading) {
        return (
            <div className='flex flex-col gap-5'>
                {[1, 2, 3].map(i => (
                    <div key={i}>
                        <ReviewSkeleton key={i}/>
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
                    <p>Summary: {currentSummary}</p>
                ) : (
                    <div>
                        <Button onClick={handleSummarize} className='cursor-pointer' disabled={isSummaryLoading}>
                            <HiSparkles />
                            Summarize
                        </Button>
                        {isSummaryLoading && <div className='py3'><ReviewSkeleton/></div>} 
                        {summaryError && <p className='text-red-500'>{summaryError}</p>}
                    </div>
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
