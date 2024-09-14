//
//  ContentView.swift
//  mnn-llm
//
//  Created by wangzhaode on 2023/12/14.
//

import SwiftUI
import Combine
import PhotosUI

class ChatViewModel: ObservableObject {
    @Published var messages: [Message] = []
    @Published var isModelLoaded = false
    @Published var isProcessing: Bool = false
    @Published var pendingImagePath: String?
    private var llm: LLMInferenceEngineWrapper?

    init() {
        self.messages.append(Message(id: UUID(), text: "qwen2-vl-2b-instruct 模型加载中, 请稍等 ...", isUser: false))
        llm = LLMInferenceEngineWrapper { [weak self] success in
            DispatchQueue.main.async {
                self?.isModelLoaded = success
                let loadResult = success ? "模型加载完毕！" : "模型加载失败！"
                self?.messages.append(Message(id: UUID(), text: loadResult, isUser: false))
            }
        }
    }

    func sendInput(_ input: String) {
        var combinedInput = input
        if let imagePath = pendingImagePath {
            combinedInput = "<img>\(imagePath)</img>" + combinedInput
        }

        let userMessage = Message(id: UUID(), text: input, imagePath: pendingImagePath, isUser: true)
        DispatchQueue.main.async {
            self.messages.append(userMessage)
            self.pendingImagePath = nil
        }
        isProcessing = true
        DispatchQueue.global(qos: .userInitiated).async {
            self.llm?.processInput(combinedInput) { [weak self] output in
                DispatchQueue.main.async {
                    if output.contains("<eop>") {
                        self?.isProcessing = false
                    } else {
                        self?.appendResponse(output)
                    }
                }
            }
        }
    }

    func saveImageToTemporaryFile(_ image: UIImage, completion: @escaping (String?) -> Void) {
        let fileManager = FileManager.default
        let tempDirectory = fileManager.temporaryDirectory
        let fileURL = tempDirectory.appendingPathComponent(UUID().uuidString).appendingPathExtension("png")
        if let imageData = image.pngData() {
            do {
                try imageData.write(to: fileURL)
                completion(fileURL.path)
            } catch {
                print("Error saving image to file: \(error)")
                completion(nil)
            }
        } else {
            completion(nil)
        }
    }

    private func appendResponse(_ output: String) {
        if let lastMessage = messages.last, !lastMessage.isUser {
            var updatedMessage = messages[messages.count - 1]
            updatedMessage.text! += output
            self.messages[messages.count - 1] = updatedMessage
        } else {
            let newMessage = Message(id: UUID(), text: output, isUser: false)
            self.messages.append(newMessage)
        }
    }

    func deleteTemporaryFile(at path: String) {
        let fileManager = FileManager.default
        do {
            try fileManager.removeItem(atPath: path)
        } catch {
            print("Error deleting temporary file: \(error)")
        }
    }
}


struct Message: Identifiable, Equatable {
    let id: UUID
    var text: String?
    var imagePath: String?
    let isUser: Bool
}

struct ChatBubble: View {
    let message: Message

    var body: some View {
        HStack {
            if message.isUser {
                Spacer()
            }

            VStack(alignment: .leading) {
                if let imagePath = message.imagePath, let image = UIImage(contentsOfFile: imagePath) {
                    Image(uiImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(maxWidth: 300, maxHeight: 200)
                        .padding(10)
                        .background(message.isUser ? Color.blue : Color.gray.opacity(0.2))
                        .cornerRadius(10)
                }

                if let text = message.text {
                    Text(text)
                        .padding(10)
                        .foregroundColor(message.isUser ? .white : .black)
                        .background(message.isUser ? Color.blue : Color.gray.opacity(0.2))
                        .cornerRadius(10)
                        .frame(maxWidth: 400, alignment: message.isUser ? .trailing : .leading)
                }
            }

            if !message.isUser {
                Spacer()
            }
        }
        .transition(.scale(scale: 0, anchor: message.isUser ? .bottomTrailing : .bottomLeading))
    }
}

struct ChatView: View {
    @StateObject var viewModel = ChatViewModel()
    @State private var inputText: String = ""
    @State private var selectedImage: UIImage?
    @State private var isImagePickerPresented: Bool = false
    @State private var isImageSourcePickerPresented: Bool = false
    @State private var imageSourceType: UIImagePickerController.SourceType = .photoLibrary

    var body: some View {
        NavigationView {
            VStack {
                ScrollView {
                    ScrollViewReader { scrollView in
                        VStack(alignment: .leading, spacing: 10) {
                            ForEach(viewModel.messages) { message in
                                ChatBubble(message: message)
                            }
                        }
                        .padding(.horizontal)
                        .onChange(of: viewModel.messages) { _ in
                            if let lastMessageId = viewModel.messages.last?.id {
                                withAnimation {
                                    scrollView.scrollTo(lastMessageId, anchor: .bottom)
                                }
                            }
                        }
                    }
                }

                HStack {
                    if let image = selectedImage {
                        Image(uiImage: image)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(width: 50, height: 50)
                            .padding()
                            .background(Color.gray.opacity(0.2))
                            .cornerRadius(10)

                        Button(action: {
                            selectedImage = nil
                        }) {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundColor(.red)
                        }
                    }

                    TextField("Type a message...", text: $inputText)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                        .frame(minHeight: 44)

                    Button(action: {
                        if let image = selectedImage {
                            viewModel.saveImageToTemporaryFile(image) { imagePath in
                                if let imagePath = imagePath {
                                    viewModel.pendingImagePath = imagePath
                                    viewModel.sendInput(inputText)
                                    inputText = ""
                                }
                            }
                        } else {
                            viewModel.sendInput(inputText)
                            inputText = ""
                        }
                        selectedImage = nil
                    }) {
                        Image(systemName: "arrow.up.circle.fill")
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(width: 44, height: 44)
                    }
                    .disabled(inputText.isEmpty && selectedImage == nil || viewModel.isProcessing || !viewModel.isModelLoaded)

                    Button(action: {
                        isImageSourcePickerPresented = true
                    }) {
                        Image(systemName: "photo.on.rectangle.angled")
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(width: 44, height: 44)
                    }
                }
                .padding()
                .actionSheet(isPresented: $isImageSourcePickerPresented) {
                    ActionSheet(
                        title: Text("Select Image Source"),
                        buttons: [
                            .default(Text("Photo Library")) {
                                imageSourceType = .photoLibrary
                                isImagePickerPresented = true
                            },
                            .default(Text("Camera")) {
                                imageSourceType = .camera
                                isImagePickerPresented = true
                            },
                            .cancel()
                        ]
                    )
                }
                .sheet(isPresented: $isImagePickerPresented) {
                    ImagePicker(selectedImage: $selectedImage, sourceType: imageSourceType)
                }
            }
            .navigationBarTitle("mnn-llm", displayMode: .inline)
        }
    }
}

struct ImagePicker: UIViewControllerRepresentable {
    @Binding var selectedImage: UIImage?
    var sourceType: UIImagePickerController.SourceType = .photoLibrary

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.delegate = context.coordinator
        picker.sourceType = sourceType
        return picker
    }

    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}

    class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        var parent: ImagePicker

        init(_ parent: ImagePicker) {
            self.parent = parent
        }

        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            if let image = info[.originalImage] as? UIImage {
                parent.selectedImage = image
            }
            picker.dismiss(animated: true)
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            picker.dismiss(animated: true)
        }
    }
}

extension String {
    var isBlank: Bool {
        return allSatisfy({ $0.isWhitespace })
    }
}

struct ChatView_Previews: PreviewProvider {
    static var previews: some View {
        ChatView()
    }
}
