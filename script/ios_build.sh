xcodebuild \
CODE_SIGN_IDENTITY="" CODE_SIGNING_REQUIRED=NO CODE_SIGN_ENTITLEMENTS="" CODE_SIGNING_ALLOWED=NO \
-project ios/mnn-llm/mnn-llm.xcodeproj \
-target mnn-llm \
-destination "platform=iOS,id=dvtdevice-DVTiPhonePlaceholder-iphoneos:placeholder,name=Any iOS Device"