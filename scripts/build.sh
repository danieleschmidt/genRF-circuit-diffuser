#!/bin/bash
# Build script for GenRF Circuit Diffuser
# Usage: ./scripts/build.sh [target] [options]

set -euo pipefail

# Configuration
PROJECT_NAME="genrf-circuit-diffuser"
REGISTRY="${DOCKER_REGISTRY:-docker.io}"
NAMESPACE="${DOCKER_NAMESPACE:-genrf}"
VERSION="${VERSION:-$(git describe --tags --always --dirty)}"
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
GIT_COMMIT=$(git rev-parse HEAD)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
GenRF Circuit Diffuser Build Script

Usage: $0 [TARGET] [OPTIONS]

TARGETS:
    development     Build development image (default)
    production      Build production image
    jupyter         Build Jupyter notebook image
    all             Build all images
    clean           Clean up build artifacts

OPTIONS:
    --push          Push images to registry after build
    --no-cache      Build without using cache
    --platform      Target platform (e.g., linux/amd64,linux/arm64)
    --registry      Docker registry (default: docker.io)
    --namespace     Docker namespace (default: genrf)
    --tag           Custom tag (default: git describe)
    --help          Show this help message

EXAMPLES:
    $0 development
    $0 production --push
    $0 all --platform linux/amd64,linux/arm64
    $0 clean

ENVIRONMENT VARIABLES:
    DOCKER_REGISTRY     Docker registry URL
    DOCKER_NAMESPACE    Docker namespace
    VERSION            Custom version tag
    DOCKER_BUILDKIT    Enable BuildKit (recommended)

EOF
}

# Parse command line arguments
TARGET="${1:-development}"
PUSH=false
NO_CACHE=""
PLATFORM=""

shift || true

while [[ $# -gt 0 ]]; do
    case $1 in
        --push)
            PUSH=true
            shift
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --platform)
            PLATFORM="--platform $2"
            shift 2
            ;;
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --tag)
            VERSION="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate requirements
check_requirements() {
    log_info "Checking build requirements..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! command -v git &> /dev/null; then
        log_error "Git is not installed or not in PATH"
        exit 1
    fi
    
    if [[ ! -f "Dockerfile" ]]; then
        log_error "Dockerfile not found in current directory"
        exit 1
    fi
    
    log_success "Requirements check passed"
}

# Build function
build_image() {
    local target=$1
    local image_name="$REGISTRY/$NAMESPACE/$PROJECT_NAME:$target-$VERSION"
    local latest_tag="$REGISTRY/$NAMESPACE/$PROJECT_NAME:$target-latest"
    
    log_info "Building $target image..."
    log_info "Image: $image_name"
    log_info "Platform: ${PLATFORM:-default}"
    
    # Build arguments
    local build_args=(
        --target "$target"
        --build-arg "BUILD_DATE=$BUILD_DATE"
        --build-arg "VERSION=$VERSION"
        --build-arg "GIT_COMMIT=$GIT_COMMIT"
        --tag "$image_name"
        --tag "$latest_tag"
    )
    
    # Add optional arguments
    if [[ -n "$NO_CACHE" ]]; then
        build_args+=("$NO_CACHE")
    fi
    
    if [[ -n "$PLATFORM" ]]; then
        build_args+=($PLATFORM)
    fi
    
    # Execute build
    if docker build "${build_args[@]}" .; then
        log_success "Successfully built $target image"
        
        # Show image size
        local size=$(docker images "$image_name" --format "table {{.Size}}" | tail -n 1)
        log_info "Image size: $size"
        
        # Push if requested
        if [[ "$PUSH" == true ]]; then
            log_info "Pushing $image_name..."
            docker push "$image_name"
            docker push "$latest_tag"
            log_success "Successfully pushed $target image"
        fi
    else
        log_error "Failed to build $target image"
        exit 1
    fi
}

# Clean function
clean_artifacts() {
    log_info "Cleaning up build artifacts..."
    
    # Remove dangling images
    docker image prune -f
    
    # Remove build cache (if BuildKit is enabled)
    if [[ "${DOCKER_BUILDKIT:-}" == "1" ]]; then
        docker builder prune -f
    fi
    
    log_success "Cleanup completed"
}

# Security scan function
security_scan() {
    local image_name=$1
    
    log_info "Running security scan on $image_name..."
    
    # Use Trivy if available
    if command -v trivy &> /dev/null; then
        trivy image --severity HIGH,CRITICAL "$image_name"
    else
        log_warning "Trivy not found, skipping security scan"
        log_info "Install with: brew install aquasec/trivy/trivy"
    fi
}

# Performance test function
performance_test() {
    local image_name=$1
    
    log_info "Running performance test on $image_name..."
    
    # Test container startup time
    local start_time=$(date +%s.%N)
    docker run --rm "$image_name" --version > /dev/null 2>&1
    local end_time=$(date +%s.%N)
    local duration=$(echo "$end_time - $start_time" | bc -l)
    
    log_info "Container startup time: ${duration}s"
}

# Main execution
main() {
    log_info "Starting GenRF Circuit Diffuser build process"
    log_info "Target: $TARGET"
    log_info "Version: $VERSION"
    log_info "Registry: $REGISTRY"
    log_info "Namespace: $NAMESPACE"
    
    check_requirements
    
    case $TARGET in
        development)
            build_image "development"
            ;;
        production)
            build_image "production"
            security_scan "$REGISTRY/$NAMESPACE/$PROJECT_NAME:production-$VERSION"
            performance_test "$REGISTRY/$NAMESPACE/$PROJECT_NAME:production-$VERSION"
            ;;
        jupyter)
            build_image "jupyter"
            ;;
        all)
            build_image "development"
            build_image "production"
            build_image "jupyter"
            ;;
        clean)
            clean_artifacts
            ;;
        *)
            log_error "Unknown target: $TARGET"
            show_help
            exit 1
            ;;
    esac
    
    log_success "Build process completed successfully!"
}

# Execute main function
main "$@"