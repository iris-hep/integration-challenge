#!/bin/bash

# Purpose: Download a file from a URL and save it based on the number before '_file_index'
# Usage: ./download_file_index.sh <URL>

# Check if URL argument is provided
if [ $# -ne 1 ]; then
    echo "Error: Missing URL argument"
    echo "Usage: $0 <URL>"
    exit 1
fi

# Store URL from argument
URL="$1"

# Check if URL contains '_file_index'
if ! echo "$URL" | grep -q "_file_index"; then
    echo "Error: URL does not contain '_file_index' pattern"
    exit 2
fi

# Extract the number before '_file_index'
NUMBER=$(echo "$URL" | grep -o '[0-9]\+_file_index' | cut -d'_' -f1)

# Check if number extraction was successful
if [ -z "$NUMBER" ]; then
    echo "Error: Could not extract number from URL"
    exit 3
fi

echo "Extracting number: $NUMBER"
echo "Downloading from: $URL"
echo "Saving as: ${NUMBER}.txt"

# Download file using curl
if curl -L "$URL" -o "${NUMBER}.txt"; then
    echo "Download successful! File saved as ${NUMBER}.txt"
else
    echo "Error: Download failed"
    exit 4
fi

