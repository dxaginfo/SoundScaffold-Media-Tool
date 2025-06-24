import os
import uuid
import json
import tempfile
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from sound_scaffold import AudioAnalyzer, AudioEnhancer, SoundLibraryManager, GeminiIntegration

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = '/tmp/sound_scaffold_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize components
audio_analyzer = AudioAnalyzer()
audio_enhancer = AudioEnhancer()
sound_library = SoundLibraryManager()
gemini_integration = GeminiIntegration()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'version': '0.2.0'
    })

@app.route('/analyze', methods=['POST'])
def analyze_audio():
    """Analyze audio file endpoint."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    temp_file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(temp_file_path)
    
    try:
        # Analyze the audio
        results = audio_analyzer.analyze(temp_file_path)
        
        # Clean up the temporary file
        os.remove(temp_file_path)
        
        return jsonify(results)
    
    except Exception as e:
        # Clean up the temporary file in case of error
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        return jsonify({'error': str(e)}), 500

@app.route('/analysis/<analysis_id>', methods=['GET'])
def get_analysis(analysis_id):
    """Get analysis results endpoint."""
    try:
        # Retrieve analysis results from Firestore
        from google.cloud import firestore
        db = firestore.Client()
        doc_ref = db.collection('audio_analyses').document(analysis_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            return jsonify({'error': 'Analysis not found'}), 404
        
        return jsonify(doc.to_dict())
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/enhance', methods=['POST'])
def enhance_audio():
    """Enhance audio file endpoint."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Get enhancement parameters
    enhancement_type = request.form.get('enhancement_type', 'dialogue_clarity')
    
    # Parse options if provided
    options = {}
    if 'options' in request.form:
        try:
            options = json.loads(request.form['options'])
        except json.JSONDecodeError:
            return jsonify({'error': 'Invalid options format'}), 400
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    temp_file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(temp_file_path)
    
    try:
        # Enhance the audio
        output_file = audio_enhancer.enhance(temp_file_path, enhancement_type, options)
        
        if not output_file:
            return jsonify({'error': 'Enhancement failed'}), 500
        
        # Upload to Cloud Storage if requested
        cloud_url = None
        if options.get('upload_to_cloud', False):
            from google.cloud import storage
            storage_client = storage.Client()
            bucket_name = options.get('bucket_name', 'sound-scaffold-assets')
            bucket = storage_client.bucket(bucket_name)
            blob_name = f"enhanced_audio/{os.path.basename(output_file)}"
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(output_file)
            blob.make_public()
            cloud_url = blob.public_url
        
        # Return result
        result = {
            'enhanced_file': os.path.basename(output_file),
            'enhancement_type': enhancement_type
        }
        
        if cloud_url:
            result['cloud_url'] = cloud_url
        
        # Clean up the temporary files
        os.remove(temp_file_path)
        
        # Keep the output file if it's in a different location
        if os.path.dirname(output_file) != UPLOAD_FOLDER:
            result['output_path'] = output_file
        else:
            os.remove(output_file)
        
        return jsonify(result)
    
    except Exception as e:
        # Clean up the temporary file in case of error
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        return jsonify({'error': str(e)}), 500

@app.route('/gemini/analyze', methods=['POST'])
def gemini_analyze_audio():
    """Analyze audio using Gemini AI endpoint."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Parse custom prompt if provided
    custom_prompt = None
    if 'prompt' in request.form:
        custom_prompt = request.form.get('prompt')
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    temp_file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(temp_file_path)
    
    try:
        # Extract audio features
        features = audio_analyzer._extract_features(temp_file_path)
        
        # Get quality metrics
        quality_metrics = audio_analyzer._assess_quality(features)
        
        # Use Gemini to analyze
        gemini_results = gemini_integration.analyze_audio_content(features, custom_prompt)
        
        # Combine results
        combined_results = {
            'file_name': os.path.basename(temp_file_path),
            'duration': features.get('duration', 0),
            'gemini_analysis': gemini_results,
            'quality_metrics': quality_metrics,
            'timestamp': features.get('timestamp')
        }
        
        # Store in Firestore
        from google.cloud import firestore
        db = firestore.Client()
        doc_ref = db.collection('gemini_analyses').document()
        doc_ref.set(combined_results)
        combined_results['analysis_id'] = doc_ref.id
        
        # Clean up the temporary file
        os.remove(temp_file_path)
        
        return jsonify(combined_results)
    
    except Exception as e:
        # Clean up the temporary file in case of error
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        return jsonify({'error': str(e)}), 500

@app.route('/gemini/enhance', methods=['POST'])
def gemini_enhancement_suggestions():
    """Get AI-powered enhancement suggestions endpoint."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    temp_file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(temp_file_path)
    
    try:
        # Extract audio features
        features = audio_analyzer._extract_features(temp_file_path)
        
        # Get quality metrics
        quality_metrics = audio_analyzer._assess_quality(features)
        
        # Get Gemini enhancement suggestions
        suggestions = gemini_integration.suggest_audio_enhancements(features, quality_metrics)
        
        # Prepare result
        result = {
            'file_name': os.path.basename(temp_file_path),
            'duration': features.get('duration', 0),
            'quality_metrics': quality_metrics,
            'enhancement_suggestions': suggestions
        }
        
        # Clean up the temporary file
        os.remove(temp_file_path)
        
        return jsonify(result)
    
    except Exception as e:
        # Clean up the temporary file in case of error
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        return jsonify({'error': str(e)}), 500

@app.route('/gemini/categorize', methods=['POST'])
def gemini_categorize_sound():
    """Categorize sound using Gemini AI endpoint."""
    # Check if it's a direct text description or a file
    if 'description' in request.form:
        description = request.form.get('description')
        audio_features = None
        
        try:
            # Categorize using text description only
            categories = gemini_integration.categorize_sound(description)
            return jsonify({
                'description': description,
                'categorization': categories
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    elif 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        temp_file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(temp_file_path)
        
        try:
            # Extract audio features
            features = audio_analyzer._extract_features(temp_file_path)
            
            # Generate description (can be overridden by user-provided description)
            description = request.form.get('description', f"Audio file: {filename}")
            
            # Categorize using Gemini
            categories = gemini_integration.categorize_sound(description, features)
            
            result = {
                'file_name': os.path.basename(temp_file_path),
                'description': description,
                'categorization': categories
            }
            
            # Clean up the temporary file
            os.remove(temp_file_path)
            
            return jsonify(result)
            
        except Exception as e:
            # Clean up the temporary file in case of error
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'No file or description provided'}), 400

@app.route('/library/search', methods=['GET'])
def search_sound_library():
    """Search sound library endpoint."""
    try:
        # Get search criteria from query parameters
        criteria = {}
        
        if 'category' in request.args:
            criteria['category'] = request.args.get('category')
        
        if 'tags' in request.args:
            criteria['tags'] = request.args.get('tags').split(',')
        
        if 'name' in request.args:
            criteria['name'] = request.args.get('name')
        
        if 'text_search' in request.args:
            criteria['text_search'] = request.args.get('text_search')
        
        if 'duration_min' in request.args:
            criteria['duration_min'] = float(request.args.get('duration_min'))
        
        if 'duration_max' in request.args:
            criteria['duration_max'] = float(request.args.get('duration_max'))
        
        # Search the library
        results = sound_library.search_sounds(criteria)
        
        return jsonify({
            'count': len(results),
            'results': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/library/add', methods=['POST'])
def add_to_library():
    """Add sound to library endpoint."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Get metadata
    metadata = {}
    if 'metadata' in request.form:
        try:
            metadata = json.loads(request.form['metadata'])
        except json.JSONDecodeError:
            return jsonify({'error': 'Invalid metadata format'}), 400
    
    # Add basic metadata from form
    for key in ['name', 'category', 'description']:
        if key in request.form:
            metadata[key] = request.form[key]
    
    # Handle tags
    if 'tags' in request.form:
        metadata['tags'] = [tag.strip() for tag in request.form['tags'].split(',')]
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    temp_file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(temp_file_path)
    
    try:
        # Add to library
        sound_id = sound_library.add_sound(temp_file_path, metadata)
        
        if not sound_id:
            return jsonify({'error': 'Failed to add sound to library'}), 500
        
        # Clean up the temporary file
        os.remove(temp_file_path)
        
        # Get sound details
        sound_details = sound_library.get_sound_details(sound_id)
        
        return jsonify({
            'sound_id': sound_id,
            'details': sound_details
        })
    
    except Exception as e:
        # Clean up the temporary file in case of error
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        return jsonify({'error': str(e)}), 500

@app.route('/scenes/<project_id>', methods=['GET'])
def get_project_scenes(project_id):
    """Get audio scenes for a project."""
    try:
        # In a real implementation, this would retrieve project-specific scene data
        # For this example, we'll return mock data
        
        from google.cloud import firestore
        db = firestore.Client()
        
        # Query analyses related to this project
        query = db.collection('audio_analyses').where('project_id', '==', project_id)
        results = query.get()
        
        scenes = []
        for doc in results:
            data = doc.to_dict()
            if 'scene_breakdown' in data:
                for scene in data['scene_breakdown']:
                    scenes.append({
                        'file_name': data.get('file_name', 'unknown'),
                        'analysis_id': doc.id,
                        **scene
                    })
        
        # If no scenes found, return mock data
        if not scenes:
            scenes = [
                {
                    'file_name': 'scene1.wav',
                    'start_time': 0,
                    'end_time': 15.5,
                    'scene_type': 'dialogue',
                    'confidence': 0.92
                },
                {
                    'file_name': 'scene1.wav',
                    'start_time': 15.5,
                    'end_time': 25.0,
                    'scene_type': 'ambient',
                    'confidence': 0.87
                },
                {
                    'file_name': 'scene2.wav',
                    'start_time': 0,
                    'end_time': 12.3,
                    'scene_type': 'music',
                    'confidence': 0.95
                }
            ]
        
        return jsonify({
            'project_id': project_id,
            'scene_count': len(scenes),
            'scenes': scenes
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)