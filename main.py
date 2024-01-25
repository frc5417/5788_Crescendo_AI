from crescendo_ai import YoLoV8_Inference

inference = YoLoV8_Inference("models/best_v8.pt")
inference.inference_stream("note.mp4")
