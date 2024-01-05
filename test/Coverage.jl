using Coverage

# カバレッジデータの読み込み
cov = process_folder(".")
# cov = process_folder("test")

# LCOV形式でファイルに出力
LCOV.writefile("coverage.lcov", cov)
