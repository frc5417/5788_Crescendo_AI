from crescendo_ai import YoLoV8_Inference

inference = YoLoV8_Inference("models/best.pt")
inference.inference_stream("note.mp4")