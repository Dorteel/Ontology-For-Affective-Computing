import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor, BlipModel, BlipProcessor
from sentence_transformers import SentenceTransformer
import faiss
import json
import cv2
from PIL import Image
from typing import List, Dict, Tuple, Optional
import pickle

class VLMKnowledgeBaseLinker:
    """
    A system for linking sensory data to knowledge bases using Vision-Language Models
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize the VLM-KB linker
        
        Args:
            model_name: HuggingFace model name for the VLM
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize VLM model and processor
        if "clip" in model_name.lower():
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
        elif "blip" in model_name.lower():
            self.model = BlipModel.from_pretrained(model_name).to(self.device)
            self.processor = BlipProcessor.from_pretrained(model_name)
        
        # Initialize text encoder for KB descriptions
        self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Storage for KB embeddings
        self.kb_embeddings = None
        self.kb_entities = []
        self.faiss_index = None
        
    # STEP 1: Knowledge Base Preprocessing
    def preprocess_knowledge_base(self, kb_file: str, save_path: str = "kb_embeddings.pkl"):
        """
        Convert knowledge base entities into embeddings
        
        Args:
            kb_file: Path to JSON file containing KB entities
            save_path: Path to save processed embeddings
        """
        print("Step 1: Preprocessing Knowledge Base...")
        
        # Load knowledge base
        with open(kb_file, 'r') as f:
            kb_data = json.load(f)
        
        entities = []
        descriptions = []
        
        for entity in kb_data['entities']:
            entities.append({
                'id': entity['id'],
                'name': entity['name'],
                'type': entity.get('type', 'unknown'),
                'attributes': entity.get('attributes', {}),
                'relationships': entity.get('relationships', []),
                'description': entity.get('description', entity['name'])
            })
            
            # Create rich text description for embedding
            desc = f"{entity['name']}. {entity.get('description', '')}"
            if 'attributes' in entity:
                attrs = ', '.join([f"{k}: {v}" for k, v in entity['attributes'].items()])
                desc += f" Attributes: {attrs}"
            descriptions.append(desc)
        
        self.kb_entities = entities
        
        # Generate embeddings using both VLM text encoder and sentence transformer
        print(f"Generating embeddings for {len(descriptions)} entities...")
        
        # VLM text embeddings
        vlm_embeddings = []
        batch_size = 32
        
        for i in range(0, len(descriptions), batch_size):
            batch_texts = descriptions[i:i+batch_size]
            
            if hasattr(self.processor, 'tokenizer'):
                inputs = self.processor.tokenizer(batch_texts, padding=True, 
                                                truncation=True, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    text_features = self.model.get_text_features(**inputs)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    vlm_embeddings.extend(text_features.cpu().numpy())
        
        # Sentence transformer embeddings for semantic richness
        st_embeddings = self.text_encoder.encode(descriptions, show_progress_bar=True)
        
        # Combine embeddings (concatenate or weighted average)
        combined_embeddings = np.concatenate([
            np.array(vlm_embeddings), st_embeddings
        ], axis=1)
        
        self.kb_embeddings = combined_embeddings
        
        # Create FAISS index for fast similarity search
        dimension = combined_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(combined_embeddings)
        self.faiss_index.add(combined_embeddings.astype('float32'))
        
        # Save processed data
        with open(save_path, 'wb') as f:
            pickle.dump({
                'embeddings': self.kb_embeddings,
                'entities': self.kb_entities,
                'faiss_index': faiss.serialize_index(self.faiss_index)
            }, f)
        
        print(f"KB preprocessing complete. Saved {len(entities)} entities to {save_path}")
    
    def load_processed_kb(self, save_path: str = "kb_embeddings.pkl"):
        """Load preprocessed KB embeddings"""
        with open(save_path, 'rb') as f:
            data = pickle.load(f)
            
        self.kb_embeddings = data['embeddings']
        self.kb_entities = data['entities']
        self.faiss_index = faiss.deserialize_index(data['faiss_index'])
    
    # STEP 2: Visual Feature Extraction
    def extract_visual_features(self, image_path: str) -> np.ndarray:
        """
        Extract visual features from image using VLM vision encoder
        
        Args:
            image_path: Path to image file
            
        Returns:
            Normalized visual feature vector
        """
        print("Step 2: Extracting Visual Features...")
        
        # Load and preprocess image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path  # Assume PIL Image
        
        # Process image through VLM
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            if hasattr(self.model, 'get_image_features'):
                image_features = self.model.get_image_features(**inputs)
            else:
                # For BLIP-style models
                image_features = self.model.vision_model(inputs['pixel_values']).last_hidden_state.mean(dim=1)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
        return image_features.cpu().numpy()
    
    # STEP 3: Similarity Search and Entity Matching
    def query_knowledge_base(self, visual_features: np.ndarray, 
                           top_k: int = 10, 
                           entity_types: Optional[List[str]] = None) -> List[Dict]:
        """
        Query KB using visual features to find matching entities
        
        Args:
            visual_features: Normalized visual feature vector
            top_k: Number of top matches to return
            entity_types: Optional filter for entity types
            
        Returns:
            List of matching entities with scores
        """
        print("Step 3: Querying Knowledge Base...")
        
        if self.faiss_index is None:
            raise ValueError("Knowledge base not loaded. Call preprocess_knowledge_base() first.")
        
        # Pad visual features to match KB embedding dimension if needed
        kb_dim = self.kb_embeddings.shape[1]
        vis_dim = visual_features.shape[1]
        
        if vis_dim < kb_dim:
            # Pad with zeros or use learned projection
            padding = np.zeros((visual_features.shape[0], kb_dim - vis_dim))
            query_features = np.concatenate([visual_features, padding], axis=1)
        else:
            query_features = visual_features[:, :kb_dim]
        
        # Normalize query features
        faiss.normalize_L2(query_features.astype('float32'))
        
        # Search FAISS index
        scores, indices = self.faiss_index.search(query_features.astype('float32'), top_k)
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            entity = self.kb_entities[idx].copy()
            entity['similarity_score'] = float(score)
            
            # Apply entity type filter
            if entity_types is None or entity['type'] in entity_types:
                results.append(entity)
        
        return results[:top_k]
    
    # STEP 4: Contextual Filtering and Scene Understanding
    def contextual_filtering(self, image_path: str, initial_matches: List[Dict], 
                           scene_context: Optional[str] = None) -> List[Dict]:
        """
        Apply contextual filtering to improve entity matching
        
        Args:
            image_path: Path to input image
            initial_matches: Initial entity matches
            scene_context: Optional scene description for context
            
        Returns:
            Filtered and re-ranked entity matches
        """
        print("Step 4: Applying Contextual Filtering...")
        
        # Generate scene description if not provided
        if scene_context is None:
            scene_context = self.generate_scene_description(image_path)
        
        # Re-rank entities based on contextual relevance
        filtered_matches = []
        
        for entity in initial_matches:
            # Calculate contextual relevance score
            context_score = self.calculate_context_relevance(entity, scene_context)
            
            # Combine with similarity score
            entity['context_score'] = context_score
            entity['combined_score'] = (entity['similarity_score'] * 0.7 + 
                                      context_score * 0.3)
            
            filtered_matches.append(entity)
        
        # Sort by combined score
        filtered_matches.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return filtered_matches
    
    def generate_scene_description(self, image_path: str) -> str:
        """Generate scene description using VLM"""
        image = Image.open(image_path).convert('RGB')
        
        # Use BLIP for image captioning if available
        if hasattr(self.model, 'generate'):
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_length=50)
                description = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        else:
            # Fallback to basic scene classification
            description = "general scene"
        
        return description
    
    def calculate_context_relevance(self, entity: Dict, scene_context: str) -> float:
        """Calculate how relevant an entity is to the scene context"""
        entity_desc = entity.get('description', entity['name'])
        
        # Use text similarity between entity description and scene context
        entity_emb = self.text_encoder.encode([entity_desc])
        context_emb = self.text_encoder.encode([scene_context])
        
        similarity = np.dot(entity_emb[0], context_emb[0]) / (
            np.linalg.norm(entity_emb[0]) * np.linalg.norm(context_emb[0])
        )
        
        return float(similarity)
    
    # STEP 5: Main Processing Pipeline
    def link_sensory_data_to_kb(self, image_path: str, 
                               top_k: int = 5,
                               entity_types: Optional[List[str]] = None,
                               use_context: bool = True) -> List[Dict]:
        """
        Complete pipeline to link sensory data to knowledge base
        
        Args:
            image_path: Path to input image
            top_k: Number of top matches to return
            entity_types: Optional filter for entity types
            use_context: Whether to apply contextual filtering
            
        Returns:
            List of linked KB entities with scores
        """
        print("=== VLM-KB Linking Pipeline ===")
        
        # Step 2: Extract visual features
        visual_features = self.extract_visual_features(image_path)
        
        # Step 3: Query knowledge base
        initial_matches = self.query_knowledge_base(
            visual_features, top_k=top_k*2, entity_types=entity_types
        )
        
        # Step 4: Apply contextual filtering
        if use_context:
            final_matches = self.contextual_filtering(image_path, initial_matches)
        else:
            final_matches = initial_matches
        
        return final_matches[:top_k]

# Example usage and demo
def create_sample_knowledge_base():
    """Create a sample knowledge base for demonstration"""
    sample_kb = {
        "entities": [
            {
                "id": "cat_001",
                "name": "Domestic Cat",
                "type": "animal",
                "description": "Small carnivorous mammal, often kept as pet",
                "attributes": {"size": "small", "habitat": "domestic", "diet": "carnivore"},
                "relationships": ["mammal", "pet", "feline"]
            },
            {
                "id": "dog_001", 
                "name": "Domestic Dog",
                "type": "animal",
                "description": "Loyal companion animal, man's best friend",
                "attributes": {"size": "variable", "habitat": "domestic", "diet": "omnivore"},
                "relationships": ["mammal", "pet", "canine"]
            },
            {
                "id": "car_001",
                "name": "Automobile",
                "type": "vehicle", 
                "description": "Motor vehicle with four wheels for transportation",
                "attributes": {"wheels": 4, "powered": "engine", "purpose": "transport"},
                "relationships": ["vehicle", "transport", "machine"]
            },
            {
                "id": "tree_001",
                "name": "Tree",
                "type": "plant",
                "description": "Large woody plant with trunk and branches",
                "attributes": {"size": "large", "habitat": "terrestrial", "type": "plant"},
                "relationships": ["plant", "nature", "woody"]
            }
        ]
    }
    
    with open("sample_kb.json", "w") as f:
        json.dump(sample_kb, f, indent=2)
    
    return "sample_kb.json"

# Demo function
def demo_vlm_kb_linking():
    """Demonstrate the VLM-KB linking system"""
    print("Creating sample knowledge base...")
    kb_file = create_sample_knowledge_base()
    
    # Initialize the linker
    linker = VLMKnowledgeBaseLinker()
    
    # Step 1: Preprocess KB
    linker.preprocess_knowledge_base(kb_file)
    
    # For demo purposes, we'll simulate image processing
    print("\n=== Demo: Linking Image to Knowledge Base ===")
    print("(In real usage, provide actual image path)")
    print("Example: results = linker.link_sensory_data_to_kb('path/to/image.jpg')")
    
    return linker

if __name__ == "__main__":
    # Run the demo
    demo_system = demo_vlm_kb_linking()