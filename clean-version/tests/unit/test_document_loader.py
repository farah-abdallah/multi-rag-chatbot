"""
Unit tests for document loading functionality.
"""

import pytest
from unittest.mock import Mock, patch, mock_open
import tempfile
import os
from pathlib import Path
from src.document.loader import DocumentLoader, PDFLoader, TextLoader, DOCXLoader
from src.utils.exceptions import DocumentLoadError


class TestDocumentLoader:
    """Test cases for DocumentLoader base class."""
    
    @pytest.fixture
    def document_loader(self):
        """Create DocumentLoader instance for testing."""
        return DocumentLoader()
    
    def test_init(self, document_loader):
        """Test DocumentLoader initialization."""
        assert document_loader.supported_extensions == []
    
    def test_load_not_implemented(self, document_loader):
        """Test that load method raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            document_loader.load("test.txt")
    
    def test_supports_extension(self, document_loader):
        """Test supports_extension method."""
        document_loader.supported_extensions = ['.txt', '.md']
        
        assert document_loader.supports_extension('.txt')
        assert document_loader.supports_extension('.md')
        assert not document_loader.supports_extension('.pdf')
    
    def test_get_metadata(self, document_loader):
        """Test get_metadata method."""
        file_path = "test.txt"
        
        with patch('os.path.getsize', return_value=1024), \
             patch('os.path.getmtime', return_value=1234567890):
            
            metadata = document_loader.get_metadata(file_path)
            
            assert metadata['source'] == file_path
            assert metadata['file_size'] == 1024
            assert metadata['last_modified'] == 1234567890
            assert 'file_type' in metadata


class TestPDFLoader:
    """Test cases for PDFLoader class."""
    
    @pytest.fixture
    def pdf_loader(self):
        """Create PDFLoader instance for testing."""
        return PDFLoader()
    
    def test_init(self, pdf_loader):
        """Test PDFLoader initialization."""
        assert '.pdf' in pdf_loader.supported_extensions
    
    @patch('PyPDF2.PdfReader')
    def test_load_pdf_success(self, mock_pdf_reader, pdf_loader):
        """Test successful PDF loading."""
        # Mock PDF reader
        mock_reader = Mock()
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "Page 1 content"
        mock_page2 = Mock()
        mock_page2.extract_text.return_value = "Page 2 content"
        mock_reader.pages = [mock_page1, mock_page2]
        mock_pdf_reader.return_value = mock_reader
        
        # Mock file operations
        with patch('builtins.open', mock_open()), \
             patch.object(pdf_loader, 'get_metadata', return_value={'source': 'test.pdf'}):
            
            documents = pdf_loader.load("test.pdf")
            
            assert len(documents) == 2
            assert documents[0]['content'] == "Page 1 content"
            assert documents[0]['metadata']['source'] == 'test.pdf'
            assert documents[0]['metadata']['page'] == 1
            assert documents[1]['content'] == "Page 2 content"
            assert documents[1]['metadata']['page'] == 2
    
    @patch('PyPDF2.PdfReader')
    def test_load_pdf_empty_pages(self, mock_pdf_reader, pdf_loader):
        """Test PDF loading with empty pages."""
        mock_reader = Mock()
        mock_page = Mock()
        mock_page.extract_text.return_value = ""
        mock_reader.pages = [mock_page]
        mock_pdf_reader.return_value = mock_reader
        
        with patch('builtins.open', mock_open()), \
             patch.object(pdf_loader, 'get_metadata', return_value={'source': 'test.pdf'}):
            
            documents = pdf_loader.load("test.pdf")
            
            assert len(documents) == 0  # Empty pages should be filtered out
    
    @patch('PyPDF2.PdfReader')
    def test_load_pdf_exception(self, mock_pdf_reader, pdf_loader):
        """Test PDF loading with exception."""
        mock_pdf_reader.side_effect = Exception("PDF read error")
        
        with patch('builtins.open', mock_open()):
            with pytest.raises(DocumentLoadError):
                pdf_loader.load("test.pdf")
    
    def test_load_unsupported_file(self, pdf_loader):
        """Test loading unsupported file type."""
        with pytest.raises(DocumentLoadError):
            pdf_loader.load("test.txt")


class TestTextLoader:
    """Test cases for TextLoader class."""
    
    @pytest.fixture
    def text_loader(self):
        """Create TextLoader instance for testing."""
        return TextLoader()
    
    def test_init(self, text_loader):
        """Test TextLoader initialization."""
        assert '.txt' in text_loader.supported_extensions
        assert '.md' in text_loader.supported_extensions
    
    def test_load_text_success(self, text_loader):
        """Test successful text file loading."""
        content = "This is a test document\nWith multiple lines"
        
        with patch('builtins.open', mock_open(read_data=content)), \
             patch.object(text_loader, 'get_metadata', return_value={'source': 'test.txt'}):
            
            documents = text_loader.load("test.txt")
            
            assert len(documents) == 1
            assert documents[0]['content'] == content
            assert documents[0]['metadata']['source'] == 'test.txt'
    
    def test_load_text_encoding_error(self, text_loader):
        """Test text file loading with encoding error."""
        with patch('builtins.open', side_effect=UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid')):
            
            # Should try different encodings
            with patch('builtins.open', mock_open(read_data="content")) as mock_file:
                documents = text_loader.load("test.txt")
                
                # Should have been called multiple times with different encodings
                assert mock_file.call_count >= 2
    
    def test_load_text_file_not_found(self, text_loader):
        """Test text file loading with file not found."""
        with patch('builtins.open', side_effect=FileNotFoundError()):
            with pytest.raises(DocumentLoadError):
                text_loader.load("nonexistent.txt")
    
    def test_load_empty_text_file(self, text_loader):
        """Test loading empty text file."""
        with patch('builtins.open', mock_open(read_data="")), \
             patch.object(text_loader, 'get_metadata', return_value={'source': 'test.txt'}):
            
            documents = text_loader.load("test.txt")
            
            assert len(documents) == 0  # Empty files should be filtered out


class TestDOCXLoader:
    """Test cases for DOCXLoader class."""
    
    @pytest.fixture
    def docx_loader(self):
        """Create DOCXLoader instance for testing."""
        return DOCXLoader()
    
    def test_init(self, docx_loader):
        """Test DOCXLoader initialization."""
        assert '.docx' in docx_loader.supported_extensions
    
    @patch('docx.Document')
    def test_load_docx_success(self, mock_document, docx_loader):
        """Test successful DOCX loading."""
        # Mock document structure
        mock_doc = Mock()
        mock_paragraph1 = Mock()
        mock_paragraph1.text = "First paragraph"
        mock_paragraph2 = Mock()
        mock_paragraph2.text = "Second paragraph"
        mock_doc.paragraphs = [mock_paragraph1, mock_paragraph2]
        mock_document.return_value = mock_doc
        
        with patch.object(docx_loader, 'get_metadata', return_value={'source': 'test.docx'}):
            documents = docx_loader.load("test.docx")
            
            assert len(documents) == 1
            assert documents[0]['content'] == "First paragraph\nSecond paragraph"
            assert documents[0]['metadata']['source'] == 'test.docx'
    
    @patch('docx.Document')
    def test_load_docx_empty_paragraphs(self, mock_document, docx_loader):
        """Test DOCX loading with empty paragraphs."""
        mock_doc = Mock()
        mock_paragraph = Mock()
        mock_paragraph.text = ""
        mock_doc.paragraphs = [mock_paragraph]
        mock_document.return_value = mock_doc
        
        with patch.object(docx_loader, 'get_metadata', return_value={'source': 'test.docx'}):
            documents = docx_loader.load("test.docx")
            
            assert len(documents) == 0  # Empty documents should be filtered out
    
    @patch('docx.Document')
    def test_load_docx_exception(self, mock_document, docx_loader):
        """Test DOCX loading with exception."""
        mock_document.side_effect = Exception("DOCX read error")
        
        with pytest.raises(DocumentLoadError):
            docx_loader.load("test.docx")


class TestLoaderFactory:
    """Test cases for loader factory functionality."""
    
    def test_get_loader_for_pdf(self):
        """Test getting PDF loader."""
        from src.document.loader import get_loader_for_file
        
        loader = get_loader_for_file("test.pdf")
        
        assert isinstance(loader, PDFLoader)
    
    def test_get_loader_for_txt(self):
        """Test getting text loader."""
        from src.document.loader import get_loader_for_file
        
        loader = get_loader_for_file("test.txt")
        
        assert isinstance(loader, TextLoader)
    
    def test_get_loader_for_docx(self):
        """Test getting DOCX loader."""
        from src.document.loader import get_loader_for_file
        
        loader = get_loader_for_file("test.docx")
        
        assert isinstance(loader, DOCXLoader)
    
    def test_get_loader_for_unsupported(self):
        """Test getting loader for unsupported file type."""
        from src.document.loader import get_loader_for_file
        
        with pytest.raises(DocumentLoadError):
            get_loader_for_file("test.xyz")
    
    def test_load_documents_multiple_files(self):
        """Test loading multiple documents."""
        from src.document.loader import load_documents
        
        files = ["test1.txt", "test2.pdf", "test3.docx"]
        
        with patch('src.document.loader.get_loader_for_file') as mock_get_loader:
            mock_loader = Mock()
            mock_loader.load.return_value = [{'content': 'test', 'metadata': {'source': 'test'}}]
            mock_get_loader.return_value = mock_loader
            
            documents = load_documents(files)
            
            assert len(documents) == 3
            assert mock_get_loader.call_count == 3
            assert mock_loader.load.call_count == 3
