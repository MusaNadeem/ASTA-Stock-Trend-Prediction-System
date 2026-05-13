import os
import re

root = 'frontend'
count = 0
for r, d, files in os.walk(root):
    for f in files:
        if f.endswith('.html') or f.endswith('.js') or f.endswith('.css'):
            filepath = os.path.join(r, f)
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Replace $ with Rs ONLY if:
            # 1. It is not followed by { (template literal)
            # 2. It is not followed by a letter (like JS variables $foo)
            new_content = re.sub(r'\$(?!\s*\{)(?![a-zA-Z])', 'Rs ', content)
            
            if new_content != content:
                with open(filepath, 'w', encoding='utf-8') as file:
                    file.write(new_content)
                count += 1
                print(f"Updated {filepath}")
print(f"Done. Updated {count} files.")