#!/usr/bin/env node

import dotenv from 'dotenv';
import OpenAI from "openai";
import { Command } from 'commander';
import readline from 'readline';

dotenv.config('./.env');
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

const client = new OpenAI({
    apiKey: OPENAI_API_KEY,
});

const program = new Command();
program
    .version('1.0.0')
    .description('Generate a git commit message based on diff content')
    .argument('[diffContent]', 'Diff content to generate a commit message from')
    .option('-f, --file <path>', 'Path to a file containing diff content')
    .action((diffContent, options) => {
        if (options.file) {
            const fs = require('fs');
            const fileContent = fs.readFileSync(options.file, 'utf-8');
            processPrompt(fileContent);
        } else if (diffContent) {
            processPrompt(diffContent);
        } else {
            const rl = readline.createInterface({
                input: process.stdin,
                output: process.stdout,
                terminal: false
            });

            let diffContent = '';

            rl.on('line', (line) => {
                diffContent += line + '\n';
            });

            rl.on('close', () => {
                if (!diffContent || diffContent.trim() === '') {
                    console.error('No diff content provided. Unable to generate a commit message without changes to summarize.');
                    process.exit(1);
                }
                processPrompt(diffContent);
            });
        }
    });

program.parse(process.argv);

async function processPrompt(prompt) {
    const systemPrompt = `You are a git commit message generator. Based on the following diff, create a concise and descriptive commit message with a max of 16 tokens:

Diff:
${prompt}

Commit Message:`;

    try {
        const stream = await client.responses.create({
            model: "gpt-4.1",
            input: systemPrompt,
            temperature: 0.5,
            max_output_tokens: 16,
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
}
