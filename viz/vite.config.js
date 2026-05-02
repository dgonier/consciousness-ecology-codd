import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SIGNALS_DIR = path.resolve(__dirname, '..', 'logs', 'signals');

export default defineConfig({
  plugins: [
    react(),
    {
      name: 'serve-signals',
      configureServer(server) {
        server.middlewares.use('/signals', (req, res) => {
          if (req.url === '/' || req.url === '') {
            try {
              const files = fs
                .readdirSync(SIGNALS_DIR)
                .filter(f => f.endsWith('.jsonl'));
              res.setHeader('Content-Type', 'application/json');
              res.end(JSON.stringify(files));
              return;
            } catch (e) {
              res.statusCode = 500;
              res.end(JSON.stringify({ error: String(e) }));
              return;
            }
          }
          const fname = decodeURIComponent(req.url.replace(/^\//, ''));
          const fpath = path.join(SIGNALS_DIR, fname);
          try {
            const data = fs.readFileSync(fpath, 'utf8');
            res.setHeader('Content-Type', 'text/plain');
            res.end(data);
          } catch (e) {
            res.statusCode = 404;
            res.end('not found');
          }
        });
      },
    },
  ],
  server: { port: 5173, host: true },
});
