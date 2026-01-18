import { defineConfig } from 'vite';
import fs from 'fs';
import path from 'path';

const IMAGE_SOURCE = '/Users/CONWARD/ideas-syncthing';

export default defineConfig({
  server: {
    port: 3000
  },
  plugins: [
    {
      name: 'serve-images',
      configureServer(server) {
        server.middlewares.use('/images', (req, res, next) => {
          try {
            const imagePath = decodeURIComponent(req.url.slice(1));
            
            // Handle both full paths and just filenames
            let fullPath;
            if (imagePath.startsWith('/')) {
              fullPath = imagePath;
            } else {
              fullPath = path.join(IMAGE_SOURCE, path.basename(imagePath));
            }
            
            if (fs.existsSync(fullPath)) {
              const ext = path.extname(fullPath).toLowerCase();
              const mimeTypes = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
              };
              res.setHeader('Content-Type', mimeTypes[ext] || 'application/octet-stream');
              fs.createReadStream(fullPath).pipe(res);
            } else {
              res.statusCode = 404;
              res.end('Image not found');
            }
          } catch (err) {
            res.statusCode = 500;
            res.end('Error serving image');
          }
        });
      }
    }
  ]
});
