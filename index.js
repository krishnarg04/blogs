document.addEventListener('DOMContentLoaded', async () => {
    try {
        // Fetch blog data
        const response = await fetch('./data/blogs.json');
        if (!response.ok) {
            throw new Error(`Failed to fetch blog data: ${response.status}`);
        }
        const blogs = await response.json();
        
        // Display blog entries on the page
        displayBlogs(blogs);
        
    } catch (error) {
        console.error('Error loading blog data:', error);
        document.getElementById('blog-list').textContent = 
            'Failed to load blog entries. Please try again later.';
    }
});

function displayBlogs(blogs) {
    const blogList = document.getElementById('blog-list');
    blogList.innerHTML = '';
    
    if (!blogs || blogs.length === 0) {
        blogList.textContent = 'No blog entries found.';
        return;
    }
    
    blogs.forEach(blog => {
        const blogEntry = document.createElement('div');
        blogEntry.className = 'blog-entry';
        
        // Create a container for text content
        const textContent = document.createElement('div');
        textContent.className = 'blog-text-content';
        
        const title = document.createElement('h2');
        title.className = 'blog-title';
        
        const link = document.createElement('a');
        // Link to post.html with the markdown file as a parameter
        const mdFileName = blog.file.replace('.html', '.md');
        link.href = `post.html?file=${mdFileName}`;
        link.textContent = blog.title;
        title.appendChild(link);
        
        const date = document.createElement('div');
        date.className = 'blog-date';
        date.textContent = new Date(blog.date).toLocaleDateString();
        
        const description = document.createElement('p');
        description.textContent = blog.description;
        
        textContent.appendChild(title);
        textContent.appendChild(date);
        textContent.appendChild(description);
        
        blogEntry.appendChild(textContent);
        
        // Add image if it exists in the blog data
        if (blog.image) {
            const imageContainer = document.createElement('div');
            imageContainer.className = 'blog-image-container';
            
            const image = document.createElement('img');
            image.src = blog.image;
            image.alt = blog.title;
            image.className = 'blog-thumbnail';
            
            imageContainer.appendChild(image);
            blogEntry.appendChild(imageContainer);
            
            // Add a class to the blog entry to indicate it has an image
            blogEntry.classList.add('blog-entry-with-image');
        }
        
        blogList.appendChild(blogEntry);
    });
}

