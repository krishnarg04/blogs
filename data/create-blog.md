# How I Built My Personal Blog Site

Creating a personal blog site has been an exciting journey, blending creativity with technical skills. In this post, I’ll walk you through the steps I took to build my blog, from planning to deployment.

## Planning the Blog

Before diving into coding, I spent some time planning the structure and design of the blog. I wanted a clean, minimalistic look with a focus on content. I decided to use **Markdown** for writing posts because of its simplicity and ease of use. Additionally, I wanted to include features like syntax highlighting for code snippets and LaTeX support for mathematical equations.

## Choosing the Tech Stack

For the front end, I chose **HTML**, **CSS**, and **JavaScript** to keep things lightweight and fast. I used the **Marked.js** library to convert Markdown files into HTML dynamically. For syntax highlighting, I integrated **Highlight.js**, which supports a wide range of programming languages. To handle LaTeX rendering, I incorporated **MathJax**, a powerful library for displaying mathematical content.

## Setting Up the Project Structure

I organized the project into the following structure:
    /root
    |   / index
    |   / post
    |   / data
    |   |   / meta files
    |   |   / data files
    |   / images


- **index.html**: The homepage listing all blog posts.
- **post.html**: The template for individual blog posts.
- **data/blogs.json**: A JSON file containing metadata for each blog post (title, date, file name).
- **data/posts/**: A folder containing Markdown files for each blog post.
- **styles/**: CSS files for styling the blog.
- **scripts/**: JavaScript files for dynamic content loading.

## Fetching and Displaying Blog Posts

The core functionality of the blog is in the `post.js` file. When a user clicks on a blog post link, the script fetches the corresponding Markdown file and converts it into HTML using **Marked.js**. It also dynamically updates the page title, post title, and date based on the metadata stored in `blogs.json`.

Here’s a snippet of the JavaScript code that handles this:

```javascript
document.addEventListener('DOMContentLoaded', async function() {
    try {
        const urlParams = new URLSearchParams(window.location.search);
        const postFile = urlParams.get('file');
        
        if (!postFile) {
            throw new Error('No post file specified');
        }

        const baseName = postFile.replace(/\.[^/.]+$/, "");
        
        const blogsResponse = await fetch('./data/blogs.json');
        const blogs = await blogsResponse.json();
        const blogMeta = blogs.find(blog => blog.file.includes(baseName));
        
        if (blogMeta) {
            document.title = blogMeta.title;
            document.getElementById('post-title').textContent = blogMeta.title;
            document.getElementById('post-date').textContent = new Date(blogMeta.date).toLocaleDateString();
        }

        const markdownResponse = await fetch(`./data/${postFile}`);
        const markdown = await markdownResponse.text();
        
        const contentElement = document.getElementById('post-content');
        contentElement.innerHTML = marked.parse(markdown);
        
        if (document.querySelectorAll('code').length > 0) {
            const highlightScript = document.createElement('script');
            highlightScript.src = 'https://cdn.jsdelivr.net/npm/highlight.js@11.7.0/lib/highlight.min.js';
            highlightScript.onload = function() {
                document.querySelectorAll('pre code').forEach((block) => {
                    hljs.highlightElement(block);
                });
            };
            document.head.appendChild(highlightScript);
            
            const highlightCSS = document.createElement('link');
            highlightCSS.rel = 'stylesheet';
            highlightCSS.href = 'https://cdn.jsdelivr.net/npm/highlight.js@11.7.0/styles/github.min.css';
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
```

## Adding LaTeX Support with MathJax

To support LaTeX equations, I integrated MathJax into the blog. MathJax allows me to write equations in LaTeX syntax, and it automatically renders them as beautifully formatted math expressions. Here’s how I set it up:
javascript
Copy

```javascript
function loadMathJax() {
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

    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js';
    script.async = true;
    document.head.appendChild(script);
}
```
## Deployment

Once the blog was ready, I deployed it using GitHub Pages. This was a straightforward process—I simply pushed the code to a GitHub repository and enabled GitHub Pages in the repository settings. The blog is now live and accessible to anyone!
Conclusion

Building this blog was a rewarding experience. It allowed me to combine my passion for writing with my technical skills. I’m excited to continue adding new features and writing more posts in the future. If you’re thinking about creating your own blog, I highly recommend giving it a try—it’s a great way to share your knowledge and showcase your work.
