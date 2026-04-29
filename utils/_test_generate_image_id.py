import pandas as pd

# Load the original CSV
df = pd.read_csv('test.csv')

# Extract the image code from the URL
df['image_id'] = df['image_link'].str.extract(r'/([A-Za-z0-9_\+\-]+)\.jpe?g', expand=False)

# Save to a new CSV
df.to_csv('test_image_id.csv', index=False)

print("New CSV saved as 'test_image_id.csv'")
