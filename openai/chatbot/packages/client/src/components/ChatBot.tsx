import { FaArrowUp } from 'react-icons/fa';
import { useForm } from 'react-hook-form';
import ReactMarkDown from 'react-markdown';

import { useEffect, useRef, useState } from 'react';

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
    const [isBotTyping, setIsBotTyping] = useState(false);
    const [error, setError] = useState('');
    const lastMessageRef = useRef<HTMLDivElement | null>(null);
    const conversationId = useRef(crypto.randomUUID());
    const { register, handleSubmit, reset, formState } = useForm<FormData>();

    useEffect(() => {
        lastMessageRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const onSubmit = async ({ prompt }: FormData) => {
        try {
            setError('');
            setMessages(prev => [...prev, { content: prompt, role: 'user' }]);
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

            setMessages(prev => [...prev, { content: data.message, role: 'bot' }]);
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

    const onCopyMessage = (event: React.ClipboardEvent) => {
        const selection = window.getSelection()?.toString().trim();
        if (selection) {
            event.preventDefault();
            event.clipboardData.setData('text/plain', selection);
        }
    }

    return (
        <div className='flex flex-col h-full'>
            <div className='flex flex-col flex-1 gap-3 mb-10 overflow-y-auto'>
                {messages.map((message, index) => (
                    <div 
                        key={index}
                        onCopy={onCopyMessage}
                        ref={index === messages.length - 1 ? lastMessageRef : null}
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
                    </div>
                ))}
                {isBotTyping && (
                    <div className='flex self-start gap-1 px-3 py-3 bg-gray-200 rounded-xl'>
                        <div className='w-2 h-2 rounded-full bg-gray-800 animate-pulse' />
                        <div className='w-2 h-2 rounded-full bg-gray-800 animate-pulse [animation-delay:0.2s]' />
                        <div className='w-2 h-2 rounded-full bg-gray-800 animate-pulse [animation-delay:0.4s]' />
                    </div>
                )}
                {error && <p className='text-red-500'>{error}</p>}
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
