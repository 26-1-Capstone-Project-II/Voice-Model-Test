import SwiftUI
import WhisperKit
import AVFoundation
struct ContentView: View {
    @State private var whisperKit: WhisperKit?
    @State private var isRecording = false
    @State private var transcription = "모델을 불러오는 중..."
    
    @State private var audioEngine = AVAudioEngine()
    @State private var audioFile: AVAudioFile?
    @State private var recordURL: URL?
    var body: some View {
        VStack(spacing: 30) {
            Text("🎙️ 발음 평가 데모 앱")
                .font(.largeTitle)
                .padding()
            Text(transcription)
                .font(.system(size: 20, weight: .bold))
                .foregroundColor(.black)
                .multilineTextAlignment(.center)
                .padding()
                .frame(minHeight: 150)
                .frame(maxWidth: .infinity)
                .background(Color.gray.opacity(0.1))
                .cornerRadius(15)
            Button(action: {
                if isRecording { stopRecording() } else { startRecording() }
            }) {
                Text(isRecording ? "녹음 중지 및 평가 시작" : "마이크 누르고 발음하기")
                    .font(.headline).foregroundColor(.white).padding()
                    .frame(maxWidth: .infinity)
                    .background(isRecording ? Color.red : Color.blue).cornerRadius(15)
            }
            .disabled(whisperKit == nil)
            .padding(.horizontal)
        }
        .onAppear { Task { await loadModel() } }
    }
    func loadModel() async {
        do {
            // 복사한 폴더명인 'Whisper_CoreML_Model'을 바라보도록 설정 완료!
            let config = WhisperKitConfig(modelFolder: "Whisper_CoreML_Model")
            whisperKit = try await WhisperKit(config)
            transcription = "✅ 준비 완료!"
        } catch {
            transcription = "❌ 로드 실패: \(error.localizedDescription)"
        }
    }
    // -- 녹음 및 평가 생략 로직 --
    func startRecording() { isRecording = true; transcription = "(녹음 중...)" }
    func stopRecording() { 
        isRecording = false; transcription = "분석 중..." 
        Task {
            let result = try? await whisperKit?.transcribe(audioPath: "임시.wav") // 실제 연결 시수정
            transcription = result?.text ?? "결과 미확인"
        }
    }
}
