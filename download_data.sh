
# Define the target directory
KAGGLE_DIR="$HOME/.kaggle"

# Check if ~/.kaggle exists
if [ ! -d "$KAGGLE_DIR" ]; then
    echo "Creating $KAGGLE_DIR and setting up Kaggle API credentials..."
    mkdir -p "$KAGGLE_DIR"
    cp kaggle.json "$KAGGLE_DIR/"
    chmod 600 "$KAGGLE_DIR/kaggle.json"
else
    echo "$KAGGLE_DIR already exists. Skipping setup."
fi

echo "Downloading datasets..."
kaggle datasets download -d manjilkarki/deepfake-and-real-images -p ./data

echo "Unzipping the dataset..."
unzip ./data/deepfake-and-real-images.zip -d ./data/deepfake-and-real-images
