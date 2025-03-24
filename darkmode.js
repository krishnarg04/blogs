/**
 * Dark Mode Toggle - A reusable component for all pages
 * Adds a toggle button for switching between light and dark mode
 * Remembers user preference using localStorage
 */

function addDarkModeToggle() {
    // Create styles for dark mode
    const style = document.createElement('style');
    style.textContent = `
        :root {
            --bg-color: #ffffff;
            --text-color: #333333;
            --heading-color: #222222;
            --header-bg: #f5f5f5;
            --code-bg: #f0f0f0;
            --link-color: #0066cc;
            --transition-speed: 0.3s;
        }
        
        body.dark-mode {
            --bg-color: #1a1a1a;
            --post-bg-color:rgba(71, 71, 71, 0.36);
            --text-color: #f0f0f0;
            --heading-color: #ffffff;
            --header-bg: #2a2a2a;
            --code-bg: #2d2d2d;
            --link-color: #5b9dd9;
        }
        
        body, #post-content, .container, main, article {
            background-color: var(--bg-color);
            color: var(--text-color);
            transition: background-color var(--transition-speed) ease, 
                        color var(--transition-speed) ease;
        }
        
        /* Ensuring headings change color in dark mode */
        h1, h2, h3, h4, h5, h6 {
            color: var(--heading-color);
            transition: color var(--transition-speed) ease;
        }
        h1 {
            background-color: var(--h1-bg-color);
        }
        
        .blog-entry {
            background-color: var(--post-bg-color);
            color: var(--text-color);
            transition: background-color var(--transition-speed) ease, 
                       color var(--transition-speed) ease;
        }

        .blog-title a {
            color: var(--text-color);
            transition: color var(--transition-speed) ease;
        }
        
        /* Ensure h2 elements within blog entries use the heading color */
        .blog-entry h2 {
            color: var(--heading-color);
            transition: color var(--transition-speed) ease;
        }
        
        /* If you need to override any default styling from another source */
        h2 {
            color: var(--heading-color) !important;
            transition: color var(--transition-speed) ease !important;
        }
        
        a {
            color: var(--link-color);
            transition: color var(--transition-speed) ease;
        }
        
        pre, code {
            background-color: var(--code-bg);
            transition: background-color var(--transition-speed) ease;
        }
        
        /* Also ensure header content changes color */
        header, header * {
            color: var(--heading-color);
            transition: color var(--transition-speed) ease;
        }
        
        .dark-mode-toggle {
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background-color: var(--header-bg);
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
            border: none;
            z-index: 1000;
            transition: background-color var(--transition-speed) ease,
                        transform 0.2s ease;
        }
        
        .dark-mode-toggle:hover {
            transform: scale(1.1);
        }
        
        .dark-mode-toggle:focus {
            outline: none;
        }
        
        .dark-mode-toggle .icon {
            font-size: 20px;
            transition: transform var(--transition-speed) ease;
        }
        
        .dark-mode-toggle .sun-icon {
            display: none;
        }
        
        .dark-mode-toggle .moon-icon {
            display: block;
        }
        
        body.dark-mode .dark-mode-toggle .sun-icon {
            display: block;
        }
        
        body.dark-mode .dark-mode-toggle .moon-icon {
            display: none;
        }
    `;
    document.head.appendChild(style);
    
    // Create the toggle button
    const toggleButton = document.createElement('button');
    toggleButton.className = 'dark-mode-toggle';
    toggleButton.setAttribute('aria-label', 'Toggle dark mode');
    toggleButton.innerHTML = `
        <span class="icon moon-icon">🌙</span>
        <span class="icon sun-icon">☀️</span>
    `;
    document.body.appendChild(toggleButton);
    
    // Check for saved preference
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
        document.body.classList.add('dark-mode');
    } else if (savedTheme === undefined && window.matchMedia('(prefers-color-scheme: dark)').matches) {
        // If no preference is set but user's system is in dark mode, use dark mode
        document.body.classList.add('dark-mode');
        localStorage.setItem('theme', 'dark');
    }
    
    // Add click event listener
    toggleButton.addEventListener('click', () => {
        document.body.classList.toggle('dark-mode');
        
        // Save preference
        if (document.body.classList.contains('dark-mode')) {
            localStorage.setItem('theme', 'dark');
        } else {
            localStorage.setItem('theme', 'light');
        }
    });
}

// Initialize dark mode toggle when DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
    addDarkModeToggle();
});