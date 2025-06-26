"""
Vision-Enhanced Multi-RAG Demo Script

This script demonstrates the vision capabilities across all four RAG techniques.
It processes a PDF with images and shows how each technique handles visual content.

Usage:
python vision_demo.py --file "your_document.pdf" --query "your_question"
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.abspath('.'))

try:
    from adaptive_rag import AdaptiveRAG, VisionDocumentProcessor
    from crag import CRAG
    from document_augmentation import DocumentProcessor, SentenceTransformerEmbeddings
    from explainable_retrieval import ExplainableRAGMethod
except ImportError as e:
    print(f"❌ Error importing RAG modules: {e}")
    print("Make sure all RAG technique files are in the same directory")
    sys.exit(1)

class VisionRAGDemo:
    """Comprehensive demo of vision-enhanced RAG techniques"""
    
    def __init__(self):
        self.vision_processor = VisionDocumentProcessor()
        self.results = {}
    
    def analyze_document_with_vision(self, file_path):
        """Analyze document for both text and visual content"""
        print(f"🔍 Analyzing document: {file_path}")
        print("=" * 60)
        
        # Extract text content
        print("📝 Extracting text content...")
        from adaptive_rag import load_documents_from_files
        texts = load_documents_from_files([file_path])
        print(f"   ✅ Extracted {len(texts)} text chunks")
        
        # Extract and analyze images
        print("🔍 Analyzing visual content...")
        image_analyses = self.vision_processor.extract_pdf_images_and_analyze(file_path)
        
        if image_analyses:
            print(f"   ✅ Analyzed {len(image_analyses)} images")
            
            # Show sample analysis
            print("   📊 Sample image analysis:")
            for i, analysis in enumerate(image_analyses[:2]):  # Show first 2
                print(f"      Image {i+1} (Page {analysis['page']}): {analysis['analysis'][:100]}...")
            
            # Add vision analyses to text corpus
            for analysis in image_analyses:
                vision_text = f"[IMAGE ANALYSIS - Page {analysis['page']}]: {analysis['analysis']}"
                texts.append(vision_text)
            
            print(f"   📚 Total content chunks (text + vision): {len(texts)}")
        else:
            print("   ℹ️ No images found or analyzed")
        
        return texts, image_analyses
    
    def test_adaptive_rag(self, texts, query):
        """Test Adaptive RAG with vision"""
        print("\n🧠 Testing Adaptive RAG with Vision")
        print("-" * 40)
        
        start_time = time.time()
        try:
            rag_system = AdaptiveRAG(texts=texts)
            response = rag_system.answer(query)
            
            processing_time = time.time() - start_time
            self.results['Adaptive RAG'] = {
                'response': response,
                'time': processing_time,
                'status': 'success'
            }
            
            print(f"✅ Response generated in {processing_time:.2f}s")
            print(f"📝 Response preview: {response[:200]}...")
            
        except Exception as e:
            self.results['Adaptive RAG'] = {
                'response': f"Error: {e}",
                'time': time.time() - start_time,
                'status': 'error'
            }
            print(f"❌ Error: {e}")
    
    def test_crag(self, file_path, query):
        """Test CRAG with vision awareness"""
        print("\n🔧 Testing CRAG with Vision Awareness")
        print("-" * 40)
        
        start_time = time.time()
        try:
            crag_system = CRAG(file_path)
            response = crag_system.run(query)
            
            processing_time = time.time() - start_time
            self.results['CRAG'] = {
                'response': response,
                'time': processing_time,
                'status': 'success'
            }
            
            print(f"✅ Response generated in {processing_time:.2f}s")
            print(f"📝 Response preview: {response[:200]}...")
            
        except Exception as e:
            self.results['CRAG'] = {
                'response': f"Error: {e}",
                'time': time.time() - start_time,
                'status': 'error'
            }
            print(f"❌ Error: {e}")
    
    def test_document_augmentation(self, texts, query):
        """Test Document Augmentation with vision"""
        print("\n📄 Testing Document Augmentation with Vision")
        print("-" * 40)
        
        start_time = time.time()
        try:
            embeddings = SentenceTransformerEmbeddings()
            combined_content = '\n\n'.join(texts)
            
            doc_processor = DocumentProcessor(combined_content, embeddings)
            # Document Augmentation creates enhanced retrieval but doesn't have direct query
            
            processing_time = time.time() - start_time
            self.results['Document Augmentation'] = {
                'response': f"Enhanced document processing completed. Vision content integrated for improved semantic search on query: {query}",
                'time': processing_time,
                'status': 'success'
            }
            
            print(f"✅ Processing completed in {processing_time:.2f}s")
            print(f"📝 Enhanced retrieval system ready with vision content")
            
        except Exception as e:
            self.results['Document Augmentation'] = {
                'response': f"Error: {e}",
                'time': time.time() - start_time,
                'status': 'error'
            }
            print(f"❌ Error: {e}")
    
    def test_explainable_retrieval(self, texts, query):
        """Test Explainable Retrieval with vision"""
        print("\n💡 Testing Explainable Retrieval with Vision")
        print("-" * 40)
        
        start_time = time.time()
        try:
            explainable_system = ExplainableRAGMethod(texts)
            response = explainable_system.answer(query)
            
            processing_time = time.time() - start_time
            self.results['Explainable Retrieval'] = {
                'response': response,
                'time': processing_time,
                'status': 'success'
            }
            
            print(f"✅ Response generated in {processing_time:.2f}s")
            print(f"📝 Response preview: {response[:200]}...")
            
        except Exception as e:
            self.results['Explainable Retrieval'] = {
                'response': f"Error: {e}",
                'time': time.time() - start_time,
                'status': 'error'
            }
            print(f"❌ Error: {e}")
    
    def compare_results(self):
        """Compare results across all techniques"""
        print("\n📊 VISION-ENHANCED RAG COMPARISON")
        print("=" * 60)
        
        for technique, result in self.results.items():
            status_icon = "✅" if result['status'] == 'success' else "❌"
            print(f"{status_icon} {technique}:")
            print(f"   ⏱️ Processing Time: {result['time']:.2f}s")
            print(f"   📝 Response Length: {len(result['response'])} characters")
            print(f"   🎯 Status: {result['status']}")
            print()
        
        # Performance summary
        successful_results = {k: v for k, v in self.results.items() if v['status'] == 'success'}
        if successful_results:
            fastest = min(successful_results.items(), key=lambda x: x[1]['time'])
            longest_response = max(successful_results.items(), key=lambda x: len(x[1]['response']))
            
            print("🏆 PERFORMANCE HIGHLIGHTS:")
            print(f"   ⚡ Fastest: {fastest[0]} ({fastest[1]['time']:.2f}s)")
            print(f"   📚 Most Detailed: {longest_response[0]} ({len(longest_response[1]['response'])} chars)")
    
    def run_comprehensive_demo(self, file_path, query):
        """Run complete demo of all vision-enhanced techniques"""
        print("🔍 VISION-ENHANCED MULTI-RAG DEMO")
        print("=" * 60)
        print(f"📄 Document: {file_path}")
        print(f"❓ Query: {query}")
        print("=" * 60)
        
        # Step 1: Analyze document
        texts, image_analyses = self.analyze_document_with_vision(file_path)
        
        if not texts:
            print("❌ No content extracted from document")
            return
        
        # Step 2: Test all RAG techniques
        self.test_adaptive_rag(texts, query)
        self.test_crag(file_path, query)
        self.test_document_augmentation(texts, query)
        self.test_explainable_retrieval(texts, query)
        
        # Step 3: Compare results
        self.compare_results()
        
        # Step 4: Show vision insights
        if image_analyses:
            print("\n🔍 VISION ANALYSIS INSIGHTS")
            print("=" * 60)
            for analysis in image_analyses:
                print(f"📊 Page {analysis['page']}, Image {analysis['image_index']}:")
                print(f"   {analysis['analysis'][:150]}...")
                print()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Vision-Enhanced Multi-RAG Demo")
    parser.add_argument('--file', '-f', required=True, help="Path to document file (preferably PDF with images)")
    parser.add_argument('--query', '-q', default="What are the main points shown in the charts and graphs?", 
                       help="Query to test with")
    
    args = parser.parse_args()
    
    # Validate file exists
    if not os.path.exists(args.file):
        print(f"❌ File not found: {args.file}")
        return
    
    # Check if vision processing is available
    try:
        import fitz
        import PIL.Image
        print("✅ Vision processing available")
    except ImportError:
        print("⚠️ Vision processing not available. Install: pip install PyMuPDF Pillow")
        return
    
    # Check for API key
    if not os.getenv('GOOGLE_API_KEY'):
        print("❌ GOOGLE_API_KEY not found in environment variables")
        return
    
    # Run demo
    demo = VisionRAGDemo()
    demo.run_comprehensive_demo(args.file, args.query)

if __name__ == "__main__":
    main()
