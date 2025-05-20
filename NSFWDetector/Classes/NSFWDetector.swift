//
//  NSFWDetector.swift
//  NSFWDetector
//
//  Created by Michael Berg on 13.08.18.
//

import Foundation
import CoreML
import Vision
import UIKit
import Atomics

@available(iOS 12.0, *)
public actor NSFWDetector {

    public static let shared = NSFWDetector()

    private var model: VNCoreMLModel?

    private init() {}
    
    public func configureCoreMLModel() async {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all
            config.allowLowPrecisionAccumulationOnGPU = true
            
            guard let model = try? model_fp8(configuration: config) else {
                print("failed to init model")
                return
            }
            
            self.model = try VNCoreMLModel(for: model.model)
        } catch let error as NSError {
            print("Model loading went wrong: \(error)")
        }
    }

    /// The Result of an NSFW Detection
    ///
    /// - error: Detection was not successful
    /// - success: Detection was successful. `nsfwConfidence`: 0.0 for safe content - 1.0 for hardcore porn ;)
    public enum DetectionResult {
        case error(Error)
        case success(nsfwConfidence: Float)
    }

    public func check(image: UIImage) async -> DetectionResult {
        let resizedImage = image.resize(to: CGSize(width: 224, height: 224))
        let requestHandler: VNImageRequestHandler?
        if let cgImage = resizedImage?.cgImage {
            requestHandler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        } else if let ciImage = resizedImage?.ciImage {
            requestHandler = VNImageRequestHandler(ciImage: ciImage, options: [:])
        } else {
            return .error(NSError(domain: "Invalid image format", code: 0, userInfo: nil))
        }
        
        guard let model = self.model else {
            return .error(NSError(domain: "Detection failed: NSFW Model initialization failed", code: 0, userInfo: nil))
        }
        
        guard let requestHandler = requestHandler else {
            return .error(NSError(domain: "either cgImage or ciImage must be set inside of UIImage", code: 0, userInfo: nil))
        }
        
        return await withCheckedContinuation { continuation in
            let hasResumed = ManagedAtomic<Bool>(false)

            let safeResume: (DetectionResult) -> Void = { result in
                if hasResumed.compareExchange(expected: false, desired: true, ordering: .relaxed).exchanged {
                    continuation.resume(returning: result)
                }
            }

            let request = VNCoreMLRequest(model: model) { request, error in
                if let error = error {
                    safeResume(.error(error))
                    return
                }
                
                let results = request.results?.first as? VNClassificationObservation
                if let identifier = results?.identifier, let confidence = results?.confidence {
                    safeResume(.success(nsfwConfidence: identifier == "NSFW" ? confidence : 1 - confidence))
                } else {
                    safeResume(.error(NSError(domain: "Detection failed: No NSFW Observation found", code: 0, userInfo: nil)))
                }
            }
            
            #if targetEnvironment(simulator)
            request.usesCPUOnly = true
            #endif
            
            do {
                try requestHandler.perform([request])
            } catch {
                safeResume(.error(error))
            }
        }
    }

    public func check(cvPixelbuffer: CVPixelBuffer, completion: @escaping (_ result: DetectionResult) -> Void) {

        let requestHandler = VNImageRequestHandler(cvPixelBuffer: cvPixelbuffer, options: [:])

        self.check(requestHandler, completion: completion)
    }
}

@available(iOS 12.0, *)
private extension NSFWDetector {

    func check(_ requestHandler: VNImageRequestHandler?, completion: @escaping (_ result: DetectionResult) -> Void) {
        
        guard let model = model else {
            completion(.error(NSError(domain: "Detection failed: NSFW Model initialization failed", code: 0, userInfo: nil)))
            return
        }

        guard let requestHandler = requestHandler else {
            completion(.error(NSError(domain: "either cgImage or ciImage must be set inside of UIImage", code: 0, userInfo: nil)))
            return
        }

        /// The request that handles the detection completion
//        let request = VNCoreMLRequest(model: self.model, completionHandler: { (request, error) in
//            guard let observations = request.results as? [VNClassificationObservation], let observation = observations.first(where: { $0.identifier == "NSFW" }) else {
//                completion(.error(NSError(domain: "Detection failed: No NSFW Observation found", code: 0, userInfo: nil)))
//
//                return
//            }
//
//            completion(.success(nsfwConfidence: observation.confidence))
//        })
        
        let request = VNCoreMLRequest(model: model) { request, error in
            let results = request.results?.first as? VNClassificationObservation
            if let identifier = results?.identifier, let confidence =  results?.confidence {
                completion(.success(nsfwConfidence: identifier == "NSFW" ? confidence : 1 - confidence))
            } else {
                completion(.error(NSError(domain: "Detection failed: No NSFW Observation found", code: 0, userInfo: nil)))
            }
            
        }
        
        /**
         * `@Required`:
         * Set to true on Simulator or VisionKit will attempt to use GPU and fail
         *
         * `@Important`:
         * Running on Simulator results in `significantly reduced accuracy`.
         * Run on physical device for acurate results
         */
        #if targetEnvironment(simulator)
        request.usesCPUOnly = true
        #endif
        
        /// Start the actual detection
        do {
            try requestHandler.perform([request])
        } catch {
            completion(.error(NSError(domain: "Detection failed: No NSFW Observation found", code: 0, userInfo: nil)))
        }
    }
}
