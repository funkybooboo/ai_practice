import { FaArrowUp } from 'react-icons/fa';
import { useForm } from 'react-hook-form';

import { useRef } from 'react';

import { Button } from './ui/button';
import axios from 'axios';

type FormData = {
    prompt: string;
};

const ChatBot = () => {
    const conversationId = useRef(crypto.randomUUID());
    const { register, handleSubmit, reset, formState } = useForm<FormData>();

    const onSubmit = async ({ prompt }: FormData) => {
        reset();
        try {
            const response = await axios.post(
                '/api/chat',
                {
                    prompt,
                    conversationId: conversationId.current,
                },
                {
                    headers: {
                        'Content-Type': 'application/json',
                    },
                }
            );

            const data = response.data;

            console.log(data);
        } catch (error) {
            console.log(error);
        }
    };

    const onKeyDown = (event: React.KeyboardEvent<HTMLFormElement>): void => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            handleSubmit(onSubmit)();
        }
    };

    return (
        <form
            onSubmit={handleSubmit(onSubmit)}
            onKeyDown={onKeyDown}
            className="flex flex-col gap-2 items-end border-2 p-4 rounded-3xl"
        >
            <textarea
                {...register('prompt', {
                    required: true,
                    validate: (data: string): boolean => data.trim().length > 0,
                })}
                className="w-full border-0 focus:outline-0 resize-none"
                placeholder="Ask anything"
                maxLength={1000}
            />
            <Button
                disabled={!formState.isValid}
                className="rounded-full w-9 h-9"
            >
                <FaArrowUp />
            </Button>
        </form>
    );
};

export default ChatBot;
