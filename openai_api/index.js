import dotenv from 'dotenv';
import OpenAI from "openai";

dotenv.config('./.env');
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

const prompt = process.argv[2];

if (!prompt || prompt.trim() === '') {
    console.error('No input provided.');
    process.exit(1);
}

const client = new OpenAI({
    apiKey: OPENAI_API_KEY,
});

// const response = await client.responses.create({
//     model: "gpt-4.1",
//     input: prompt,
//     temperature: 0.7,
//     max_output_tokens: 50,
// });
//
// console.log(response);

try {
    const stream = await client.responses.create({
        model: "gpt-4.1",
        input: prompt,
        temperature: 0.7,
        max_output_tokens: 250,
        stream: true,
    });

    for await (const event of stream) {
        if (event.delta) {
            process.stdout.write(event.delta);
        }
    }
} catch (error) {
    console.error('Error calling OpenAI API:', error);
}
