import os
import json
import time
from main import CogniPDFNexus

def run_integration_tests():
    nexus = CogniPDFNexus()
    test_dir = "tests"
    
    test_cases = [
        {
            "name": "Academic_Research",
            "config": "tests/input/sample_config_academic.json",
            "expected": "tests/expected_output/academic_test_output.json"
        },
        {
            "name": "Business_Analysis",
            "config": "tests/input/sample_config_business.json",
            "expected": "tests/expected_output/business_test_output.json"
        }
    ]
    
    for test in test_cases:
        print(f"\n🚀 Running Test: {test['name']}")
        
        # Load config
        with open(test['config']) as f:
            config = json.load(f)
        
        # Process documents
        start_time = time.time()
        output = nexus.process(os.path.join(test_dir, "input"), config)
        duration = time.time() - start_time
        
        # Validate output structure
        assert "metadata" in output
        assert "extracted_sections" in output
        assert "sub_section_analysis" in output
        
        # Save results
        os.makedirs(os.path.join(test_dir, "output"), exist_ok=True)
        output_path = os.path.join(test_dir, "output", f"{test['name']}_output.json")
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
            
        # Basic validation
        print(f"⏱️ Processing Time: {duration:.2f}s")
        print(f"📄 Sections Found: {len(output['extracted_sections'])}")
        print(f"🔍 Subsections Found: {len(output['sub_section_analysis'])}")
        print(f"💾 Output saved to: {output_path}")
        
        print(f"✅ Test Completed: {test['name']}")
    
    print("\n🎉 All integration tests completed!")

if __name__ == "__main__":
    run_integration_tests()import os
import json
import time
from main import CogniPDFNexus

def run_integration_tests():
    nexus = CogniPDFNexus()
    test_dir = "tests"
    
    test_cases = [
        {
            "name": "Academic_Research",
            "config": "tests/input/sample_config_academic.json",
            "expected": "tests/expected_output/academic_test_output.json"
        },
        {
            "name": "Business_Analysis",
            "config": "tests/input/sample_config_business.json",
            "expected": "tests/expected_output/business_test_output.json"
        }
    ]
    
    for test in test_cases:
        print(f"\n🚀 Running Test: {test['name']}")
        
        # Load config
        with open(test['config']) as f:
            config = json.load(f)
        
        # Process documents
        start_time = time.time()
        output = nexus.process(os.path.join(test_dir, "input"), config)
        duration = time.time() - start_time
        
        # Validate output structure
        assert "metadata" in output
        assert "extracted_sections" in output
        assert "sub_section_analysis" in output
        
        # Save results
        os.makedirs(os.path.join(test_dir, "output"), exist_ok=True)
        output_path = os.path.join(test_dir, "output", f"{test['name']}_output.json")
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
            
        # Basic validation
        print(f"⏱️ Processing Time: {duration:.2f}s")
        print(f"📄 Sections Found: {len(output['extracted_sections'])}")
        print(f"🔍 Subsections Found: {len(output['sub_section_analysis'])}")
        print(f"💾 Output saved to: {output_path}")
        
        print(f"✅ Test Completed: {test['name']}")
    
    print("\n🎉 All integration tests completed!")

if __name__ == "__main__":
    run_integration_tests()