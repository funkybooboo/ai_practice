import { Router } from 'express';

import chatController from './controllers/chat.controller';

const router = Router();

router.post('/api/chat', chatController.sendMessage);

export default router;
