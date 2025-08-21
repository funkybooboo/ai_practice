import { FaArrowUp } from 'react-icons/fa';
import { useForm } from 'react-hook-form';
import ReactMarkDown from 'react-markdown';

import { useRef, useState } from 'react';

import { Button } from './ui/button';
import axios from 'axios';

type FormData = {
    prompt: string;
};

type ChatReponse = {
    message: string;
};

type Message = {
    content: string;
    role: 'user' | 'bot';
};

const ChatBot = () => {
    const [messages, setMessages] = useState<Message[]>([]);
    const conversationId = useRef(crypto.randomUUID());
    const { register, handleSubmit, reset, formState } = useForm<FormData>();

    const onSubmit = async ({ prompt }: FormData) => {
        setMessages(prev => [...prev, { content: prompt, role: 'user' }]);

        reset();

        try {
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

            setMessages(prev => [...prev, { content: data.message, role: 'bot' }]);
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
        <div>
            <div className='flex flex-col gap-3 mb-10'>
                {messages.map((message, index) => (
                    <p 
                        key={index}
                        className={`
                            px-3 py-1 rounded-xl
                            ${
                                message.role === 'user' ? 
                                'bg-blue-600 text-white self-end' : 
                                'bg-gray-100 text-black self-start'
                            }
                        `}
                    >
                        <ReactMarkDown>
                            {message.content}
                        </ReactMarkDown>
                    </p>
                ))}
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
