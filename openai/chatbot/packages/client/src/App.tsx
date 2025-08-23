import { useState } from 'react';
import ChatBot from './components/chat/ChatBot';
import ReviewList from './components/reviews/ReviewList';
import { Button } from './components/ui/button';

function App() {
    const [activeTab, setActiveTab] = useState('chat');

    const handleTabChange = (tab: string) => {
        setActiveTab(tab);
    };

    return (
        <div className="h-screen flex flex-col">
            <div className="flex space-x-4 p-4 bg-gray-100 shadow-md">
                <Button
                    className={`px-4 py-2 cursor-pointer ${activeTab === 'chat' ? 'bg-black' : 'bg-gray-500'}`}
                    onClick={() => handleTabChange('chat')}
                >
                    Chat
                </Button>
                <Button
                    className={`px-4 py-2 cursor-pointer ${activeTab === 'reviews' ? 'bg-black' : 'bg-gray-500'}`}
                    onClick={() => handleTabChange('reviews')}
                >
                    Reviews
                </Button>
            </div>
            <div className="flex-1 p-4 overflow-auto">
                {activeTab === 'chat' ? (
                    <ChatBot />
                ) : (
                    <ReviewList productId={4} />
                )}
            </div>
        </div>
    );
}

export default App;
