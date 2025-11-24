import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import glob

class ReverseImageSearch:
    def __init__(self, model_name='vit'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_name == 'vit':
            self.model = self.load_vit_model()
        else:
            self.model = self.load_resnet_model()
        
        self.transform = self.get_transforms()
        self.image_embeddings = {}
    
    def load_vit_model(self):
        """Load Vision Transformer model"""
        try:
            # Using torchvision's ViT
            model = models.vit_b_32(pretrained=True)
            model.heads = torch.nn.Identity()  # Remove classification head
            model.eval()
            model.to(self.device)
            return model
        except:
            print("ViT model not available, falling back to ResNet")
            return self.load_resnet_model()
    
    def load_resnet_model(self):
        """Load ResNet model as fallback"""
        model = models.resnet50(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove last layer
        model.eval()
        model.to(self.device)
        return model
    
    def get_transforms(self):
        """Get image transforms for the model"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def extract_embedding(self, image_path):
        """Extract embedding from an image"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model(image_tensor)
                embedding = embedding.squeeze().cpu().numpy()
            
            return embedding
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def create_image_database(self, image_folder):
        """Create database of image embeddings"""
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_paths = []
        
        for extension in image_extensions:
            image_paths.extend(glob.glob(os.path.join(image_folder, extension)))
            image_paths.extend(glob.glob(os.path.join(image_folder, extension.upper())))
        
        print(f"Found {len(image_paths)} images in {image_folder}")
        
        for image_path in image_paths:
            embedding = self.extract_embedding(image_path)
            if embedding is not None:
                self.image_embeddings[image_path] = embedding
        
        print(f"Successfully processed {len(self.image_embeddings)} images")
    
    def search_similar_images(self, query_image_path, top_k=5):
        """Search for similar images"""
        query_embedding = self.extract_embedding(query_image_path)
        if query_embedding is None:
            return []
        
        similarities = []
        for img_path, embedding in self.image_embeddings.items():
        
            query_emb_2d = query_embedding.reshape(1, -1)
            db_emb_2d = embedding.reshape(1, -1)
            
            similarity = cosine_similarity(query_emb_2d, db_emb_2d)[0][0]
            similarities.append((img_path, similarity))
        
    
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def search_similar_images_from_pil(self, pil_image, top_k=5):
        """Search using PIL Image object instead of file path"""
        try:
    
            image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                query_embedding = self.model(image_tensor)
                query_embedding = query_embedding.squeeze().cpu().numpy()
            
            similarities = []
            for img_path, embedding in self.image_embeddings.items():
                query_emb_2d = query_embedding.reshape(1, -1)
                db_emb_2d = embedding.reshape(1, -1)
                
                similarity = cosine_similarity(query_emb_2d, db_emb_2d)[0][0]
                similarities.append((img_path, similarity))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            print(f"Error processing PIL image: {e}")
            return []

def demo_reverse_image_search():

    ris = ReverseImageSearch(model_name='vit')
    
    image_folder = "path/to/your/images"
    ris.create_image_database(image_folder)
    
    query_image = "path/to/query/image.jpg"
    similar_images = ris.search_similar_images(query_image, top_k=5)
    
    print("Most similar images:")
    for i, (img_path, similarity) in enumerate(similar_images, 1):
        print(f"{i}. {os.path.basename(img_path)} - Similarity: {similarity:.4f}")

