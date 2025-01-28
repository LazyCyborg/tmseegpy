const fs = require('fs-extra');
const path = require('path');


async function cleanBuild() {
    const paths = [
        path.join(__dirname, '..', 'release'),
        path.join(__dirname, '..', 'dist'),
        path.join(__dirname, '..', 'python-embedded')
    ];

    console.log('Cleaning previous build artifacts...');

    for (const p of paths) {
        try {
            await fs.remove(p);
            console.log(`Cleaned: ${p}`);
        } catch (error) {
            console.warn(`Warning while cleaning ${p}:`, error.message);
        }
    }
}

cleanBuild()
    .then(() => console.log('Cleanup completed'))
    .catch(error => {
        console.error('Fatal error during cleanup:', error);
        process.exit(1);
    });

cleanBuild().catch(console.error);