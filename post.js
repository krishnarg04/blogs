document.addEventListener('DOMContentLoaded', async function() {
    try {
        // Get the filename from the URL
        const urlParams = new URLSearchParams(window.location.search);
        const postFile = urlParams.get('file');
        
        if (!postFile) {
            throw new Error('No post file specified');
        }

        // Extract the base name without extension for lookup in blogs.json
        const baseName = postFile.replace(/\.[^/.]+$/, "");
        
        // Fetch the blog metadata to get the title and date
        const blogsResponse = await fetch('./data/blogs.json');
        if (!blogsResponse.ok) {
            throw new Error('Failed to fetch blog metadata');
        }
        
        const blogs = await blogsResponse.json();
        const blogMeta = blogs.find(blog => blog.file.includes(baseName));
        
        if (blogMeta) {
            document.title = blogMeta.title;
            document.getElementById('post-title').textContent = blogMeta.title;
            document.getElementById('post-date').textContent = new Date(blogMeta.date).toLocaleDateString();
        }

        // Fetch the markdown file
        const markdownResponse = await fetch(`./data/${postFile}`);
        if (!markdownResponse.ok) {
            throw new Error(`Failed to load post: ${markdownResponse.status}`);
        }
        
        const markdown = await markdownResponse.text();
        
        // Load MathJax for LaTeX support
        loadMathJax();
        
        // Convert markdown to HTML and insert into the page
        const contentElement = document.getElementById('post-content');
        contentElement.innerHTML = marked.parse(markdown);
        
        // Process images to set max width and height
        processImages();
        
        // Add syntax highlighting if code blocks are present
        if (document.querySelectorAll('code').length > 0) {
            const highlightScript = document.createElement('script');
            highlightScript.src = 'https://cdn.jsdelivr.net/npm/highlight.js@11.7.0/lib/highlight.min.js';
            highlightScript.onload = function() {
                document.querySelectorAll('pre code').forEach((block) => {
                    hljs.highlightElement(block);
                });
            };
            document.head.appendChild(highlightScript);
            
            // Use a dark theme for highlight.js that works well with dark backgrounds
            const highlightCSS = document.createElement('link');
            highlightCSS.rel = 'stylesheet';
            highlightCSS.href = 'https://cdn.jsdelivr.net/npm/highlight.js@11.7.0/styles/atom-one-dark.min.css';
            document.head.appendChild(highlightCSS);
        }
    } catch (error) {
        console.error('Error loading blog post:', error);
        document.getElementById('post-content').innerHTML = `
            <div class="error">
                <h2>Error Loading Post</h2>
                <p>${error.message}</p>
                <p><a href="index.html">Return to blog homepage</a></p>
            </div>
        `;
    }
});

function processImages() {
    const MAX_WIDTH = 800;  // Maximum width in pixels
    const MAX_HEIGHT = 600; // Maximum height in pixels
    
    // Select all images and GIFs in the post content
    const mediaElements = document.querySelectorAll('#post-content img');
    
    mediaElements.forEach(element => {
        // Apply max dimensions as inline styles
        element.style.maxWidth = `${MAX_WIDTH}px`;
        element.style.maxHeight = `${MAX_HEIGHT}px`;
        element.style.height = 'auto';
        element.style.display = 'block';
        element.style.margin = '20px auto';
        
        // Add proper handling for GIFs
        if (element.src.toLowerCase().endsWith('.gif')) {
            // Optional: add specific styling for GIFs if needed
            element.classList.add('gif-media');
        }
        
        // Optional: add loading="lazy" for better performance
        element.loading = 'lazy';
        
        // Handle errors gracefully
        element.onerror = function() {
            this.onerror = null;
            this.src = './images/placeholder.png'; // Replace with your placeholder image path
            this.alt = 'Image failed to load';
        };
    });
}

// Function to load MathJax
function loadMathJax() {
    // Add MathJax configuration
    window.MathJax = {
        tex: {
            inlineMath: [['$', '$'], ['\\(', '\\)']],
            displayMath: [['$$', '$$'], ['\\[', '\\]']],
            processEscapes: true
        },
        svg: {
            fontCache: 'global'
        },
        options: {
            renderActions: {
                // Force re-render when the DOM updates
                addMenu: [],
                checkLoading: []
            }
        },
        startup: {
            pageReady() {
                return MathJax.startup.defaultPageReady().then(() => {
                    console.log('MathJax initialized');
                });
            }
        }
    };

    // Create script element for MathJax
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js';
    script.async = true;
    document.head.appendChild(script);
}
