# for each "src/x.cu", generate target "x"
file(GLOB_RECURSE all_srcs *.cu)
foreach(v ${all_srcs})
    string(REGEX MATCH "src/.*" relative_path ${v})
    # message(${relative_path})
    get_filename_component(target_name ${v} NAME)
    string(REGEX REPLACE ".cu" "" target_name ${target_name})

    add_executable(${target_name} ${v})
endforeach()