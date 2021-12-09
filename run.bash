#!/bin/bash
set -e

PROJECT_NAME="aoc-2021"

APP_DIR="/${PROJECT_NAME}"
IMAGE_PREFIX="${PROJECT_NAME}-ci"

function docker_image_exists() {
    local image="$1"; shift
    docker inspect "$image" >/dev/null 2>&1
}

function image_needs_to_be_rebuilt() {
    local image="$1"; shift
    [[ "$FORCE_REBUILD" == "true" ]] || ! docker_image_exists "$image"
}

function resolve_image() {
    local image="$1"; shift

    case "$image" in
    "$IMAGE_PREFIX")
        local dockerfile="Dockerfile"
        local hash="$(git hash-object "$dockerfile" $(git ls-files -- '*requirements.txt') | sha1sum | awk '{print$1}')"
        image="${image}:${hash}"
        if image_needs_to_be_rebuilt "$image"; then
            docker build -t "$image" -f "$dockerfile" .
        fi
        printf '%s' "$image"
        ;;
    *)
        echo "Unknown image '$image'" >&2
        return 1
        ;;
    esac
}

function run_in_docker() {
    local image="$1"; shift
    local command="$1"; shift
    image="$(resolve_image "$image")"
    local flags=()
    flags=("${flags[@]}" --rm) # remove container after use
    flags=("${flags[@]}" -t) # attach to terminal
    flags=("${flags[@]}" -v "$(pwd):${APP_DIR}") # mount working directory
    flags=("${flags[@]}" -w "${APP_DIR}") # set working directory to mounted local working directory
    docker run "${flags[@]}" "$image" /bin/bash -c "$command"
}

run_in_docker "$IMAGE_PREFIX" "$@"
