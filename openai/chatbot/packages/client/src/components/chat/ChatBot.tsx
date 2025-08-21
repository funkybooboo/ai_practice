import { FaArrowUp } from 'react-icons/fa';
import { useForm } from 'react-hook-form';
import axios from 'axios';

import { useRef, useState } from 'react';

import { Button } from '../ui/button';

import TypingIndicator from './TypingIndicator';
import type { Message } from './ChatMessages';
import ChatMessages from './ChatMessages';

type FormData = {
    prompt: string;
};

type ChatReponse = {
    message: string;
};

const ChatBot = () => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [isBotTyping, setIsBotTyping] = useState(false);
    const [error, setError] = useState('');
    
    const conversationId = useRef(crypto.randomUUID());
    const { register, handleSubmit, reset, formState } = useForm<FormData>();

    const onSubmit = async ({ prompt }: FormData) => {
        try {
            setError('');
            setMessages((prev) => [...prev, { content: prompt, role: 'user' }]);
            setIsBotTyping(true);

            reset({ prompt: '' });

            const { data } = await axios.post<ChatReponse>(
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

            setMessages((prev) => [
                ...prev,
                { content: data.message, role: 'bot' },
            ]);
        } catch (error) {
            console.log(error);
            setError('Something went wrong, try again!');
        } finally {
            setIsBotTyping(false);
        }
    };

    const onKeyDown = (event: React.KeyboardEvent): void => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            handleSubmit(onSubmit)();
        }
    };

    return (
        <div className="flex flex-col h-full">
            <div className="flex flex-col flex-1 gap-3 mb-10 overflow-y-auto">
                <ChatMessages messages={messages} />
                {isBotTyping && <TypingIndicator />}
                {error && <p className="text-red-500">{error}</p>}
            </div>

            <form
                onSubmit={handleSubmit(onSubmit)}
                onKeyDown={onKeyDown}
                className="flex flex-col gap-2 items-end border-2 p-4 rounded-3xl"
            >
                <textarea
                    {...register('prompt', {
                        required: true,
                        validate: (data: string): boolean =>
                            data.trim().length > 0,
                    })}
                    autoFocus
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
        </div>
    );
};

export default ChatBot;
