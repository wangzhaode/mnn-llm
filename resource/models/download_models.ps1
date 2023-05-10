function help {
    Write-Host "model download script"
    Write-Host "Usage: ./download_model.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "    -h,--help                   Print this help message and exit"
    Write-Host "    fp16,int8,int4              Chose different models"
    Write-Host "    proxy                       Use https://ghproxy.com to proxy github when download file"
}

function fp16_model {
    mkdir fp16
    for ($i = 0; $i -le 27; $i++) {
        Invoke-WebRequest -Uri "$($args[0])https://github.com/wangzhaode/ChatGLM-MNN/releases/download/v0.1/glm_block_$i.mnn" -OutFile "fp16\glm_block_$i.mnn"
    }

    Invoke-WebRequest -Uri "$($args[0])https://github.com/wangzhaode/ChatGLM-MNN/releases/download/v0.1/lm.mnn" -OutFile "fp16\lm.mnn"
    Invoke-WebRequest -Uri "$($args[0])https://github.com/wangzhaode/ChatGLM-MNN/releases/download/v0.1/slim_word_embeddings.bin" -OutFile "fp16\slim_word_embeddings.bin"
}

function int8_model {
    mkdir int8
    for ($i = 0; $i -le 27; $i++) {
        Invoke-WebRequest -Uri "$($args[0])https://github.com/wangzhaode/ChatGLM-MNN/releases/download/v0.2/glm_block_$i.mnn" -OutFile "int8\glm_block_$i.mnn"
    }

    Invoke-WebRequest -Uri "$($args[0])https://github.com/wangzhaode/ChatGLM-MNN/releases/download/v0.1/lm.mnn" -OutFile "int8\lm.mnn"
    Invoke-WebRequest -Uri "$($args[0])https://github.com/wangzhaode/ChatGLM-MNN/releases/download/v0.1/slim_word_embeddings.bin" -OutFile "int8\slim_word_embeddings.bin"
}

function int4_model {
    mkdir int4
    for ($i = 0; $i -le 27; $i++) {
        Invoke-WebRequest -Uri "$($args[0])https://github.com/wangzhaode/ChatGLM-MNN/releases/download/v0.3/glm_block_$i.mnn" -OutFile "int4\glm_block_$i.mnn"
    }

    Invoke-WebRequest -Uri "$($args[0])https://github.com/wangzhaode/ChatGLM-MNN/releases/download/v0.1/lm.mnn" -OutFile "int4\lm.mnn"
    Invoke-WebRequest -Uri "$($args[0])https://github.com/wangzhaode/ChatGLM-MNN/releases/download/v0.1/slim_word_embeddings.bin" -OutFile "int4\slim_word_embeddings.bin"
}

if ($args.Length -eq 0) {
    help
    exit 1
}

$proxy = $false

foreach ($arg in $args) {
    switch ($arg) {
        "--help" {
            help
        }
        "-h" {
            help
        }
        "proxy" {
            $proxy = $true
        }
    }
}

if ($proxy) {
    foreach ($arg in $args) {
        switch ($arg) {
            "fp16" {
                fp16_model "https://ghproxy.com/"
            }
            "int8" {
                int8_model "https://ghproxy.com/"
            }
            "int4" {
                int4_model "https://ghproxy.com/"
            }
            default {
                help
            }
        }
    }
} else {
    foreach ($arg in $args) {
        switch ($arg) {
            "fp16" {
                fp16_model ""
            }
            "int8" {
                int8_model ""
            }
            "int4" {
                int4_model ""
            }
            default {
                help
            }
        }
    }
}