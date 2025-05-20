// swift-tools-version: 5.10

import PackageDescription

let package = Package(
    name: "NSFWDetector",
    platforms: [.iOS(.v15)],
    products: [
        .library(
            name: "NSFWDetector",
            targets: ["NSFWDetector"]
        )
    ],
    dependencies: [
        .package(
            url: "https://github.com/apple/swift-atomics.git",
            from: "1.1.0"
        )
    ], targets: [
        .target(
            name: "NSFWDetector",
            dependencies: [
                .product(name: "Atomics", package: "swift-atomics")
            ],
            path: "NSFWDetector"
        )
    ]
)
