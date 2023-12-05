Param(
    [String]$model
)

function model_test($model) {
    Write-Output "test model : ${model}"
    powershell .\script\download_model.ps1 ${model}
    cd build
    .\Release\cli_demo ..\${model} prompt.txt
    cd ..
}

function test_all() {
    model_test chatglm-6b
    model_test chatglm2-6b
    model_test chatglm3-6b
    model_test codegeex2-6b
    model_test qwen-7b-chat
    model_test baichuan2-7b-chat
    model_test llama2-7b-chat
}

if ($model -eq "all") {
    test_all
} else {
    model_test $model
}