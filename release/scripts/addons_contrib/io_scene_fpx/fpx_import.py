# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

# <pep8 compliant>

###############################################################################
#234567890123456789012345678901234567890123456789012345678901234567890123456789
#--------1---------2---------3---------4---------5---------6---------7---------


# ##### BEGIN COPYRIGHT BLOCK #####
#
# initial script copyright (c)2013 Alexander Nussbaumer
#
# ##### END COPYRIGHT BLOCK #####


#import python stuff
import io
from mathutils import (
        Euler,
        Vector,
        Matrix,
        )
from math import (
        radians,
        )
from os import (
        path,
        listdir,
        rmdir,
        remove,
        )
from sys import (
        exc_info,
        )
from time import (
        time,
        )


# import io_scene_fpx stuff
if repr(globals()).find("bpy") != -1:
    from io_scene_fpx.fpx_strings import (
            fpx_str,
            )
    from io_scene_fpx.fpx_spec import (
            Fpm_File_Reader,
            Fpl_File_Reader,
            Fpt_File_Reader,
            Fpm_Model_Type,
            Fpl_Library_Type,
            FptElementType,
            Fpt_PackedLibrary_Type,
            )
    from io_scene_fpx.fpx_ui import (
            FpxUI,
            )
    from io_scene_fpx.fpx_utils import (
            FpxUtilities,
            )
else:
    from fpx_strings import (
            fpx_str,
            )
    from fpx_spec import (
            Fpm_File_Reader,
            Fpl_File_Reader,
            Fpt_File_Reader,
            Fpm_Model_Type,
            Fpl_Library_Type,
            FptElementType,
            Fpt_PackedLibrary_Type,
            )
    from fpx_ui import (
            FpxUI,
            )
    from fpx_utils import (
            FpxUtilities,
            )


#import blender stuff
from bpy import (
        ops,
        app,
        #data,
        )
import bmesh
from bpy_extras.image_utils import (
        load_image,
        )


###############################################################################
FORMAT_SCENE = "{}.s"
FORMAT_GROUP = "{}.g"
FORMAT_IMAGE = "{}.i"
FORMAT_TEXTURE = "{}.tex"
# keep material name like it is (prevent name "snakes" on re-import)
#FORMAT_MATERIAL = "{}.mat"
FORMAT_MATERIAL = "{}"
FORMAT_ACTION = "{}.act"
FORMAT_MESH = "{}.m"
FORMAT_MESH_OBJECT = "{}.mo"
FORMAT_EMPTY_OBJECT = "{}.eo"
FORMAT_DUPLI_OBJECT = "{}.do"
FORMAT_CURVE = "{}.c"
FORMAT_CURVE_OBJECT = "{}.co"
FORMAT_ARMATURE = "{}.a"
FORMAT_ARMATURE_OBJECT = "{}.ao"
FORMAT_ARMATURE_NLA = "{}.an"

#FORMAT_RESOURCE = "{}\\{}"
FORMAT_RESOURCE = "{{{}}}.{}"

PREFIX_LOCAL = "1"
PREFIX_EMBEDDED = "0"

FORMAT_MODEL_SECONDARY = "{}.low"
FORMAT_MODEL_MASK = "{}.mask"
FORMAT_MODEL_MIRROR = "{}.mirror"
FORMAT_MODEL_START = "{}.start"
FORMAT_MODEL_END = "{}.end"
FORMAT_MODEL_CAP1 = "{}.cap1"
FORMAT_MODEL_CAP2 = "{}.cap2"
FORMAT_MODEL_RING = "{}.ring{:02d}"

TRANSLITE_OBJECT = 2
###############################################################################
class FpmImporter():
    """ Load a Future Pinball Model FPM File """
    LAYERS_PRIMARY_MODEL = (
            True, False, False, False, False,
            False, False, False, False, False,
            False, False, False, False, False,
            False, False, False, False, False
            )
    LAYERS_SECONDARY_MODEL = (
            False, True, False, False, False,
            False, False, False, False, False,
            False, False, False, False, False,
            False, False, False, False, False
            )
    LAYERS_MASK_MODEL = (
            False, False, True, False, False,
            False, False, False, False, False,
            False, False, False, False, False,
            False, False, False, False, False
            )
    LAYERS_REFLECTION_MODEL = (
            False, False, False, True, False,
            False, False, False, False, False,
            False, False, False, False, False,
            False, False, False, False, False
            )
    LAYERS_COLLISION_MODEL = (
            False, False, False, False, True,
            False, False, False, False, False,
            False, False, False, False, False,
            False, False, False, False, False
            )

    def __init__(self,
            report,
            verbose=FpxUI.PROP_DEFAULT_VERBOSE,
            keep_temp=FpxUI.PROP_DEFAULT_KEEP_TEMP,
            use_all_models_of_folder=FpxUI.PROP_DEFAULT_ALL_MODELS,
            use_scene_per_model=FpxUI.PROP_DEFAULT_SCENE,
            name_extra=FpxUI.PROP_DEFAULT_NAME_EXTRA,
            use_model_filter=FpxUI.PROP_DEFAULT_USE_MODEL_FILTER,
            use_model_adjustment=FpxUI.PROP_DEFAULT_MODEL_ADJUST_FPM,
            keep_name=False,
            ):
        self.report = report
        self.verbose = verbose
        self.keep_temp = keep_temp
        self.use_all_models_of_folder = use_all_models_of_folder
        self.use_scene_per_model = use_scene_per_model
        self.name_extra = name_extra
        self.use_model_filter = use_model_filter
        self.use_model_adjustment = use_model_adjustment
        self.keep_name = keep_name

    def read(self, blender_context, filepath):
        """ read fpm file and convert fpm content to bender content """
        t1 = time()
        t2 = None

        fpx_reader = None

        self.__context = blender_context
        self.__blend_data = blender_context.blend_data

        try:
            self.folder_name, file_name = path.split(filepath)

            debug_data = []

            files = None
            if self.use_all_models_of_folder:
                files = [path.join(self.folder_name, f) for f in listdir(self.folder_name) if f.endswith(".fpm")]
            else:
                files = [filepath, ]

            for file in files:
                self.folder_name, file_name = path.split(filepath)
                try:
                    with io.FileIO(file, 'rb') as raw_io:
                        # read and inject fpm data from disk to internal structure
                        fpx_reader = Fpm_File_Reader(raw_io)
                        fpx_reader.read_model()
                        raw_io.close()
                finally:
                    pass

                # if option is set, this time will enlarges the io time
                #if self.verbose and reader:
                #    fpx_reader.print_internal()
                t2 = time()

                if fpx_reader:
                    temp_name = path.join(app.tempdir, "__grab__fpm__")
                    dst_path, dst_sub_path_names = fpx_reader.grab_content(temp_name)

                    model_name = fpx_reader.PinModel.get_value("name")
                    model_name = FpxUtilities.toGoodName(model_name) ####
                    #model_name = path.split(file)[1]
                    #model_name = model_name.lower()
                    #if model_name.endswith(".fpm"):
                    #    model_name = model_name[:-4]

                    if self.name_extra:
                        model_name = FORMAT_RESOURCE.format(self.name_extra, model_name)


                    model_filepath = dst_sub_path_names.get("modeldata")
                    debug_data.append("type={}, model_filepath='{}'".format(dst_sub_path_names.get("type"), model_filepath))
                    if model_filepath:
                        self.read_ex(blender_context, dst_sub_path_names, model_name, debug_data)

                    # cleanup
                    if not self.keep_temp:
                        try:
                            rmdir(dst_path)
                        except:
                            pass

            if self.verbose in FpxUI.VERBOSE_NORMAL:
                print()
                print("##########################################################")
                print("Import from FPM to Blender")
                for item in debug_data:
                    print("#DEBUG", item)
                print("##########################################################")

        except Exception as ex:
            type, value, traceback = exc_info()
            if self.verbose in FpxUI.VERBOSE_NORMAL:
                print("read fpm - exception in try block\n  type: '{0}'\n"
                        "  value: '{1}'".format(type, value, traceback))

            if t2 is None:
                t2 = time()

            raise

        else:
            pass

        finally:
            self.__context = None
            self.__blend_data = None

        t3 = time()
        if self.verbose in FpxUI.VERBOSE_NORMAL:
            print(fpx_str['SUMMARY_IMPORT'].format(
                    (t3 - t1), (t2 - t1), (t3 - t2)))

        return {"FINISHED"}

    ###########################################################################
    def read_ex(self, blender_context, dst_sub_path_names, model_name, debug_data):
        model_filepath = dst_sub_path_names.get("primary_model_data")
        if model_filepath:
            if self.use_scene_per_model:
                blender_scene = blender_context.blend_data.scenes.new(FORMAT_SCENE.format(model_name))
                blender_context.screen.scene = blender_scene
            else:
                blender_scene = blender_context.scene
                # setup current Scene to default units
                FpxUtilities.set_scene_to_default(blender_scene)

            blender_scene.layers = self.LAYERS_PRIMARY_MODEL
            #{'FINISHED'}
            #{'CANCELLED'}
            if 'FINISHED' in ops.import_scene.ms3d(filepath=model_filepath, use_animation=True):
                name = blender_context.active_object.name
                src_ext = "ms3d"
                index = name.rfind(".{}.".format(src_ext))
                if index < 0:
                    index = name.rfind(".")
                    #if index < 0:
                    #    return

                src_name = "{}.{}".format(name[:index], src_ext)

                remove_material(blender_context)
                if not self.keep_name:
                    rename_active_ms3d(blender_context, src_name, model_name)

                if self.use_model_adjustment:
                    adjust_position(blender_context, blender_scene, dst_sub_path_names)

                if FpxUI.USE_MODEL_FILTER_SECONDARY in self.use_model_filter:
                    model_filepath = dst_sub_path_names.get("secondary_model_data")
                    if model_filepath:
                        blender_scene.layers = self.LAYERS_SECONDARY_MODEL
                        if 'FINISHED' in ops.import_scene.ms3d(filepath=model_filepath, use_animation=False):
                            remove_material(blender_context)
                            if not self.keep_name:
                                rename_active_ms3d(blender_context, src_name, model_name, "secondary")

                if FpxUI.USE_MODEL_FILTER_MASK in self.use_model_filter:
                    model_filepath = dst_sub_path_names.get("mask_model_data")
                    if model_filepath:
                        blender_scene.layers = self.LAYERS_MASK_MODEL
                        if 'FINISHED' in ops.import_scene.ms3d(filepath=model_filepath, use_animation=False):
                            remove_material(blender_context)
                            if not self.keep_name:
                                rename_active_ms3d(blender_context, src_name, model_name, "mask")

                if FpxUI.USE_MODEL_FILTER_REFLECTION in self.use_model_filter:
                    model_filepath = dst_sub_path_names.get("reflection_model_data")
                    if model_filepath:
                        blender_scene.layers = self.LAYERS_REFLECTION_MODEL
                        if 'FINISHED' in ops.import_scene.ms3d(filepath=model_filepath, use_animation=False):
                            remove_material(blender_context)
                            if not self.keep_name:
                                rename_active_ms3d(blender_context, src_name, model_name, "reflection")

                if FpxUI.USE_MODEL_FILTER_COLLISION in self.use_model_filter:
                    ## TODO
                    pass

                blender_scene.layers = self.LAYERS_PRIMARY_MODEL

        # setup all current 3d Views of the current scene to metric units
        FpxUtilities.set_scene_to_metric(blender_context)

        # cleanup
        if not self.keep_temp:
            for key, file in dst_sub_path_names.items():
                if key in {'type', 'sub_dir', }:
                    continue
                try:
                    remove(file)
                except:
                    pass

            sub_dir_path = dst_sub_path_names.get('sub_dir')
            if sub_dir_path:
                try:
                    rmdir(sub_dir_path)
                except:
                    pass



###############################################################################
class FplImporter():
    """ Load a Future Pinball Library FPL File """
    def __init__(self,
            report,
            verbose=FpxUI.PROP_DEFAULT_VERBOSE,
            keep_temp=FpxUI.PROP_DEFAULT_KEEP_TEMP,
            use_all_libraries_of_folder=FpxUI.PROP_DEFAULT_ALL_LIBRARIES,
            use_library_filter=FpxUI.PROP_DEFAULT_USE_LIBRARY_FILTER,
            use_model_filter=FpxUI.PROP_DEFAULT_USE_MODEL_FILTER,
            use_model_adjustment=FpxUI.PROP_DEFAULT_MODEL_ADJUST_FPL,
            keep_name=False,
            ):
        self.report = report
        self.verbose = verbose
        self.keep_temp = keep_temp
        self.use_all_libraries_of_folder = use_all_libraries_of_folder
        self.use_library_filter = use_library_filter
        self.use_model_filter = use_model_filter
        self.use_model_adjustment = use_model_adjustment
        self.keep_name = keep_name

    def read(self, blender_context, filepath):
        """ read fpl file and convert fpm content to bender content """
        t1 = time()
        t2 = None

        fpx_reader = None

        self.__context = blender_context
        self.__blend_data = blender_context.blend_data
        active_scene = self.__context.screen.scene

        try:
            self.folder_name, file_name = path.split(filepath)

            debug_data = []

            if self.use_all_libraries_of_folder:
                files = [path.join(self.folder_name, f) for f in listdir(self.folder_name) if f.endswith(".fpl")]
            else:
                files = [filepath, ]

            for file in files:
                self.folder_name, file_name = path.split(file)
                try:
                    with io.FileIO(file, 'rb') as raw_io:
                        # read and inject fpl data from disk to internal structure
                        fpx_reader = Fpl_File_Reader(raw_io)
                        fpx_reader.read_library()
                        raw_io.close()
                finally:
                    pass

                # if option is set, this time will enlarges the io time
                #if self.verbose and reader:
                #    fpx_reader.print_internal()
                t2 = time()

                if fpx_reader:
                    temp_name = path.join(app.tempdir, "__grab__fpl__")
                    dst_path, dst_sub_path_names = fpx_reader.grab_content(temp_name)

                    for key, item in dst_sub_path_names.items():
                        if key is not None and key.startswith('type_'):
                            type = item
                            key_name = key[5:]
                            #print("#DEBUG", key_name, type)

                            if type not in self.use_library_filter:
                                continue

                            item_path = dst_sub_path_names.get(key_name)

                            if type == Fpl_Library_Type.TYPE_MODEL:
                                #print("#DEBUG", type, key_name)
                                FpmImporter(
                                        report=self.report,
                                        verbose=self.verbose,
                                        keep_temp=self.keep_temp,
                                        use_scene_per_model=True,
                                        name_extra=file_name,
                                        use_model_filter=self.use_model_filter,
                                        use_model_adjustment=self.use_model_adjustment,
                                    ).read(
                                            blender_context=self.__context,
                                            filepath=item_path,
                                        )
                                if not self.keep_name:
                                    rename_active_fpm(self.__context, FORMAT_RESOURCE.format(file_name, key_name))

                            elif type == Fpl_Library_Type.TYPE_GRAPHIC:
                                #print("#DEBUG", type, key_name)
                                blend_image = self.__blend_data.images.load(item_path)
                                blend_image.name = FpxUtilities.toGoodName(FORMAT_RESOURCE.format(file_name, FORMAT_IMAGE.format(key_name)))
                                blend_image.pack()
                                blend_image.use_fake_user = True
                                item_dir, item_file = path.split(item_path)
                                blend_image.filepath_raw = "//unpacked_resource/{}".format(item_file)

                        else:
                            pass


                    # cleanup
                    if not self.keep_temp:
                        cleanup_sub_dirs = []

                        for key, file in dst_sub_path_names.items():
                            if key is not None and key.startswith('sub_dir'):
                                cleanup_sub_dirs.append(file)
                                continue

                            if key in {'type', None, } or key.startswith('type'):
                                continue

                            try:
                                remove(file)
                            except:
                                pass

                        sub_dir_path = dst_sub_path_names.get('sub_dir')
                        if sub_dir_path:
                            try:
                                rmdir(sub_dir_path)
                            except:
                                pass

                        for sub_dir_path in cleanup_sub_dirs:
                            try:
                                rmdir(sub_dir_path)
                            except:
                                pass

                        try:
                            rmdir(dst_path)
                        except:
                            pass

            if self.verbose in FpxUI.VERBOSE_NORMAL:
                print()
                print("##########################################################")
                print("Import from FPM to Blender")
                for item in debug_data:
                    print("#DEBUG", item)
                print("##########################################################")

        except Exception as ex:
            type, value, traceback = exc_info()
            if self.verbose in FpxUI.VERBOSE_NORMAL:
                print("read fpl - exception in try block\n  type: '{0}'\n"
                        "  value: '{1}'".format(type, value, traceback))

            if t2 is None:
                t2 = time()

            raise

        else:
            pass

        finally:
            self.__context.screen.scene = active_scene
            self.__context = None
            self.__blend_data = None

        t3 = time()
        if self.verbose in FpxUI.VERBOSE_NORMAL:
            print(fpx_str['SUMMARY_IMPORT'].format(
                    (t3 - t1), (t2 - t1), (t3 - t2)))

        return {"FINISHED"}

    ###########################################################################


###############################################################################
class FptImporter():
    """ Load a Future Pinball Table FPT File """
    LAYERS_WIRE_RING = (
            True, True, False, False, True,
            False, False, False, False, False,
            False, False, True, False, False,
            False, False, False, False, False
            )
    LAYERS_LIGHT_SPHERE = (
            True, True, False, False, True,
            False, False, False, False, False,
            True, False, False, False, False,
            False, False, False, False, False
            )
    BLENDER_OBJECT_NAME = 0

    def __init__(self,
            report,
            verbose=FpxUI.PROP_DEFAULT_VERBOSE,
            keep_temp=FpxUI.PROP_DEFAULT_KEEP_TEMP,
            path_libraries=FpxUI.PROP_DEFAULT_LIBRARIES_PATH,
            convert_to_mesh=FpxUI.PROP_DEFAULT_CONVERT_TO_MESH,
            resolution_wire_bevel=FpxUI.PROP_DEFAULT_RESOLUTION_WIRE_BEVEL,
            resolution_wire=FpxUI.PROP_DEFAULT_RESOLUTION_WIRE,
            resolution_rubber_bevel=FpxUI.PROP_DEFAULT_RESOLUTION_RUBBER_BEVEL,
            resolution_rubber=FpxUI.PROP_DEFAULT_RESOLUTION_RUBBER,
            resolution_shape=FpxUI.PROP_DEFAULT_RESOLUTION_SHAPE,
            use_hermite_handle=FpxUI.PROP_DEFAULT_USE_HERMITE_HANDLE,
            use_library_filter=FpxUI.PROP_DEFAULT_USE_LIBRARY_FILTER,
            use_model_filter=FpxUI.PROP_DEFAULT_USE_MODEL_FILTER,
            use_model_adjustment=FpxUI.PROP_DEFAULT_MODEL_ADJUST_FPT,
            keep_name=False,
            ):
        self.report = report
        self.verbose = verbose
        self.keep_temp = keep_temp
        self.path_libraries = path_libraries
        self.convert_to_mesh = convert_to_mesh
        self.resolution_wire_bevel = resolution_wire_bevel
        self.resolution_wire = resolution_wire
        self.resolution_rubber_bevel = resolution_rubber_bevel
        self.resolution_rubber = resolution_rubber
        self.resolution_shape = resolution_shape
        self.use_hermite_handle = use_hermite_handle
        self.use_library_filter = use_library_filter
        self.use_model_filter = use_model_filter
        self.use_model_adjustment = use_model_adjustment
        self.keep_name = keep_name

        self.blend_resource_file = get_blend_resource_file_name()

        self.debug_light_extrude = 0.2
        self.debug_lightball_size = 2.0
        self.debug_lightball_height = 4.5
        self.debug_create_full_ramp_wires = False
        self.debug_missing_resources = set()

    ###########################################################################
    # create empty blender fp_table
    # read fpt file
    # fill blender with fp_table content
    def read(self, blender_context, filepath, ):
        """ read fpt file and convert fpt content to bender content """
        t1 = time()
        t2 = None

        fpx_reader = None

        self.__context = blender_context
        self.__blend_data = blender_context.blend_data

        try:
            try:
                with io.FileIO(filepath, 'rb') as raw_io:
                    # read and inject fpt data from disk to internal structure
                    fpx_reader = Fpt_File_Reader(raw_io)
                    fpx_reader.read_table()
                    raw_io.close()
            finally:
                pass

            # if option is set, this time will enlarges the io time
            #if self.options.verbose and reader:
            #    fpx_reader.print_internal()

            t2 = time()
            if fpx_reader:
                temp_name = path.join(app.tempdir, "__grab__fpt__")
                dst_path, dst_sub_path_names = fpx_reader.grab_content(temp_name)

                # setup current Scene to default units
                ##FpxUtilities.set_scene_to_default(self.__context.scene)

                self.folder_name, file_name = path.split(filepath)

                # search linked libraries
                self.fpx_images = {}
                self.GetLinked(fpx_reader.Image, self.fpx_images, Fpt_PackedLibrary_Type.TYPE_IMAGE, dst_sub_path_names)

                self.fpx_pinmodels = {}
                self.GetLinked(fpx_reader.PinModel, self.fpx_pinmodels, Fpt_PackedLibrary_Type.TYPE_MODEL, dst_sub_path_names)

                for key, item in self.fpx_images.items():
                    print("#DEBUG image:", key, item)

                for key, item in self.fpx_pinmodels.items():
                    print("#DEBUG pinmodel:", key, item)

                # build pincab
                self.CreatePinCab(fpx_reader.Table_Data)

                # handle table elements
                for key, fpx_item in fpx_reader.Table_Element.items():
                    if fpx_item:
                        object_appers_on = fpx_item.get_value("object_appers_on")
                        if object_appers_on == TRANSLITE_OBJECT:
                            continue
                        #print("#DEBUG", object_appers_on, key, fpx_item)

                        fpx_item_name = fpx_item.get_value("name")
                        fpx_item_name = FpxUtilities.toGoodName(fpx_item_name) ####
                        if not fpx_item_name:
                            continue

                        fpx_id = fpx_item.get_obj_value("id")

                        ## get the height level (wall and/or surface) on what the item will be placed
                        fpx_surface_name = fpx_item.get_value("surface")
                        fpx_surface = None
                        fpx_position_z = None
                        fpx_position_zw = None
                        if fpx_surface_name:
                            fpx_wall = fpx_reader.Walls.get(FpxUtilities.toGoodName(fpx_surface_name))
                            if fpx_wall:
                                fpx_position_zw = fpx_wall.get_value("height")
                                fpx_surface_name = fpx_wall.get_value("surface")
                                if fpx_surface_name:
                                    fpx_surface = fpx_reader.Walls.get(FpxUtilities.toGoodName(fpx_surface_name))
                            else:
                                fpx_surface = fpx_reader.Surfaces.get(FpxUtilities.toGoodName(fpx_surface_name))
                            if fpx_surface:
                                fpx_position_z = fpx_surface.get_value("top_height")

                        if fpx_position_zw is None:
                            fpx_position_zw = 0.0
                        if fpx_position_z is None:
                            fpx_position_z = 0.0

                        fpx_position_z += fpx_position_zw

                        fpx_offset = fpx_item.get_value("offset")
                        if fpx_offset:
                            fpx_position_z += fpx_offset

                        ## gather common information
                        blender_object = None
                        fpx_shape_points = fpx_item.get_value("shape_point")
                        fpx_ramp_points =  fpx_item.get_value("ramp_point")
                        fpx_position_xy = fpx_item.get_value("position")
                        fpx_render_object = fpx_item.get_value("render_object")
                        fpx_transparency = fpx_item.get_value("transparency")
                        fpx_layer = fpx_item.get_value("layer")
                        fpx_sphere_mapping = fpx_item.get_value("sphere_mapping")
                        fpx_crystal = fpx_item.get_value("crystal")
                        fpx_base = None # TODO:

                        layers = self.FpxLayerToBlenderLayers(fpx_layer, fpx_id, fpx_render_object, fpx_transparency, fpx_sphere_mapping, fpx_crystal, fpx_base)

                        # handle curve objects with shape_points
                        if fpx_shape_points:
                            if fpx_id == FptElementType.SURFACE:
                                blender_object = self.CreateSurface(fpx_item_name, layers, fpx_shape_points, fpx_item.get_value("top_height"), fpx_item.get_value("bottom_height"))
                            elif fpx_id == FptElementType.LIGHT_SHAPEABLE:
                                blender_object = self.CreateLightShapeable(fpx_item_name, layers, fpx_shape_points, fpx_position_z)
                            elif fpx_id == FptElementType.RUBBER_SHAPEABLE:
                                blender_object = self.CreateRubberShapeable(fpx_item_name, layers, fpx_shape_points, fpx_position_z)
                            elif fpx_id == FptElementType.GUIDE_WALL:
                                blender_object = self.CreateGuideWall(fpx_item_name, layers, fpx_shape_points, fpx_position_z, fpx_item.get_value("height"), fpx_item.get_value("width"))
                            elif fpx_id == FptElementType.GUIDE_WIRE:
                                blender_object = self.CreateGuideWire(fpx_item_name, layers, fpx_shape_points, fpx_position_z, fpx_item.get_value("height"), fpx_item.get_value("width"))
                            else:
                                blender_object = None
                        # handle curve objects with ramp_points
                        elif fpx_ramp_points:
                            if fpx_id == FptElementType.RAMP_WIRE:
                                blender_object = self.CreateWireRamp(fpx_item_name, layers, fpx_ramp_points, fpx_position_z, fpx_item.get_value("start_height"), fpx_item.get_value("end_height"), fpx_id, fpx_item.get_value("model_start"), fpx_item.get_value("model_end"))
                            elif fpx_id == FptElementType.RAMP_RAMP:
                                blender_object = self.CreateRamp(fpx_item_name, layers, fpx_ramp_points, fpx_position_z, fpx_item.get_value("start_height"), fpx_item.get_value("end_height"), fpx_item.get_value("start_width"), fpx_item.get_value("end_width"), fpx_item.get_value("left_side_height"), fpx_item.get_value("right_side_height"))
                            else:
                                blender_object = None
                        else:
                            if fpx_id == FptElementType.LIGHT_LIGHTIMAGE:
                                blender_object = self.CreateLightImage(fpx_item_name, layers, fpx_position_xy, fpx_position_z, fpx_item.get_value("height"), fpx_item.get_value("width"), fpx_item.get_value("rotation"))
                            else:
                                blender_object = None

                        # put the just created object (curve) to its layer
                        #if blender_object:
                        #    blender_object.layers = layers

                        if fpx_position_xy:
                            fpx_rotation = fpx_item.get_value("rotation")
                            if fpx_rotation:
                                blender_rotation = Euler((0.0, 0.0, radians(self.angle_correction(fpx_rotation))), 'XZY')
                            else:
                                blender_rotation = Euler((0.0, 0.0, 0.0), 'XZY')

                            if fpx_id in {FptElementType.CONTROL_FLIPPER, FptElementType.CONTROL_DIVERTER, }:
                                fpx_start_angle = fpx_item.get_value("start_angle")
                                if fpx_start_angle is None:
                                    fpx_start_angle = 0
                                m0 = blender_rotation.to_matrix()
                                m1 = Euler((0.0, 0.0, radians(self.angle_correction(fpx_start_angle))), 'XZY').to_matrix()
                                blender_rotation = (m0 * m1).to_euler('XZY')

                            blender_position = Vector(self.geometry_correction((fpx_position_xy[0], fpx_position_xy[1], fpx_position_z)))

                            blender_empty_object = self.__blend_data.objects.new(FORMAT_EMPTY_OBJECT.format(fpx_item_name), None)
                            blender_empty_object.location = blender_position
                            blender_empty_object.rotation_mode = 'XZY'
                            blender_empty_object.rotation_euler = blender_rotation
                            blender_empty_object.empty_draw_type = 'ARROWS'
                            blender_empty_object.empty_draw_size = 10.0
                            self.__context.scene.objects.link(blender_empty_object)
                            blender_empty_object.layers = layers

                            blender_empty_object.fpt.name = fpx_item_name

                            # handle model object (try to create an instance of existing group)
                            fpx_model_name = fpx_item.get_value("model")
                            if fpx_model_name:
                                fpx_model_beam_width = fpx_item.get_value("beam_width")
                                if fpx_model_beam_width:
                                    offset = 3.75
                                    self.attach_dupli_group(blender_empty_object, layers, fpx_model_name, "model", Vector((0.0 , ((fpx_model_beam_width / 2.0) + offset), 0.0)), -90)
                                    self.attach_dupli_group(blender_empty_object, layers, fpx_model_name, "model", Vector((0.0 , -((fpx_model_beam_width / 2.0) + offset), 0.0)), 90)
                                else:
                                    self.attach_dupli_group(blender_empty_object, layers, fpx_model_name, "model")

                            fpx_model_name_cap = fpx_item.get_value("model_cap")
                            if fpx_model_name_cap:
                                self.attach_dupli_group(blender_empty_object, layers, fpx_model_name_cap, "model_cap")

                            fpx_model_name_base = fpx_item.get_value("model_base")
                            if fpx_model_name_base:
                                self.attach_dupli_group(blender_empty_object, layers, fpx_model_name_base, "model_base")
                                self.attach_dupli_group(blender_empty_object, layers, "bumperring", 'LOCAL', Vector((0.0 , 0.0, 4.0)))
                                self.attach_dupli_group(blender_empty_object, layers, "bumperskirt", 'LOCAL', Vector((0.0 , 0.0, 3.5)))

                            fpx_model_name_start = fpx_item.get_value("model_start")
                            if fpx_model_name_start:
                                self.attach_dupli_group(blender_empty_object, layers, fpx_model_name_start, "model_start")

                            fpx_model_name_end = fpx_item.get_value("model_end")
                            if fpx_model_name_end:
                                self.attach_dupli_group(blender_empty_object, layers, fpx_model_name_end, "model_end")

                            if fpx_id == FptElementType.RUBBER_ROUND:
                                blender_object = self.CreateRubberRound(fpx_item_name, layers, fpx_position_xy, fpx_position_z, fpx_item.get_value("subtype"))

                            if fpx_id:
                                blender_empty_object.fpt.id = FptElementType.VALUE_INT_TO_NAME.get(fpx_id)

                            if fpx_id == FptElementType.LIGHT_ROUND:
                                blender_object = self.CreateLightRound(fpx_item_name, layers, fpx_position_xy, fpx_position_z, fpx_item.get_value("diameter"))
                                if blender_object:
                                    blender_object.layers = layers

                            ## #DEBUG : light dummies
                            if fpx_id in FptElementType.SET_LIGHT_OBJECTS: # and not blender_empty_object.children:
                                if ops.mesh.primitive_ico_sphere_add.poll():
                                    blender_object = ops.mesh.primitive_ico_sphere_add(subdivisions=2, size=self.debug_lightball_size, location=blender_empty_object.location + Vector((0.0, 0.0, self.debug_lightball_height)), layers=FptImporter.LAYERS_LIGHT_SPHERE)

                    # cleanup
                    if not self.keep_temp:
                        cleanup_sub_dirs = []

                        for key, file in dst_sub_path_names.items():
                            if key is not None and key.startswith('sub_dir'):
                                cleanup_sub_dirs.append(file)
                                continue

                            if key in {'type', 'data', None, } or key.startswith('type') or key.startswith('data'):
                                continue

                            try:
                                remove(file)
                            except:
                                pass

                        sub_dir_path = dst_sub_path_names.get('sub_dir')
                        if sub_dir_path:
                            try:
                                rmdir(sub_dir_path)
                            except:
                                pass

                        for sub_dir_path in cleanup_sub_dirs:
                            try:
                                rmdir(sub_dir_path)
                            except:
                                pass

                        try:
                            rmdir(dst_path)
                        except:
                            pass

                # setup all current 3d Views of the current scene to metric units
                FpxUtilities.set_scene_to_metric(self.__context)

            if self.verbose in FpxUI.VERBOSE_NORMAL:
                print()
                print("##########################################################")
                print("Import from FPT to Blender")
                print("##########################################################")

        except Exception as ex:
            type, value, traceback = exc_info()
            if self.verbose in FpxUI.VERBOSE_NORMAL:
                print("read fpt - exception in try block\n  type: '{0}'\n"
                        "  value: '{1}'".format(type, value, traceback))

            if t2 is None:
                t2 = time()

            raise

        else:
            pass

        finally:
            self.__context = None
            self.__blend_data = None

        t3 = time()
        if self.verbose in FpxUI.VERBOSE_NORMAL:
            print(fpx_str['SUMMARY_IMPORT'].format(
                    (t3 - t1), (t2 - t1), (t3 - t2)))

        return {"FINISHED"}

    ###########################################################################
    def CreateSurface(self, name, layers, fpx_points, top, bottom):
        obj, cu, act_spline = self.CreateCurve(name, layers, self.resolution_shape)

        modifier_edge_split = obj.modifiers.new("edge_split", type='EDGE_SPLIT')

        if top is None:
            top = 0.0
        if bottom is None:
            bottom = 0.0
        cu.extrude = (top - bottom) / 2.0
        cu.dimensions = '2D'

        act_spline.use_cyclic_u = True
        self.CreateCurvePoints(act_spline, fpx_points)

        obj.location = Vector((obj.location.x, obj.location.y, (top - cu.extrude)))
        return obj

    def CreateRubberShapeable(self, name, layers, fpx_points, surface):
        obj, cu, act_spline = self.CreateCurve(name, layers, self.resolution_shape)

        bevel_name = "__fpx_rubber_shapeable_bevel__"
        rubber_bevel = self.__blend_data.objects.get(bevel_name)
        if rubber_bevel is None:
            if ops.curve.primitive_bezier_circle_add.poll():
                ops.curve.primitive_bezier_circle_add()
                rubber_bevel = self.__context.active_object
                rubber_bevel.name = bevel_name
                rubber_bevel.data.dimensions = '2D'
                rubber_bevel.data.resolution_u = self.resolution_rubber
                rubber_bevel.data.splines[0].resolution_u = self.resolution_rubber
                scale = 2.4
                rubber_bevel.scale = Vector((scale,scale,scale))
        cu.bevel_object = rubber_bevel

        offset = 2.5
        act_spline.use_cyclic_u = True
        self.CreateCurvePoints(act_spline, fpx_points, (surface + offset))

        return obj

    def CreateRubberRound(self, name, layers, position_xy, surface, subtype):
        #diameter = [44, 18.5, 13.5, 12, ]
        diameter = [13.5, 18.5, 12, 44, ]

        bevel_name = "__fpx_guide_rubber_bevel__"
        wire_bevel = self.__blend_data.objects.get(bevel_name)
        if wire_bevel is None:
            cu0 = self.__blend_data.curves.new(bevel_name, 'CURVE')
            wire_bevel = self.__blend_data.objects.new(bevel_name, cu0)
            self.__context.scene.objects.link(wire_bevel)
            cu0.dimensions = '2D'
            cu0.resolution_u = self.resolution_rubber_bevel

            h = 'AUTO'
            cu0.splines.new('BEZIER')
            p0 = Vector((0.0, 0.0, 0.0))
            s0 = 5.0 / 2.0
            spline0 = cu0.splines[-1]
            spline0.resolution_u = self.resolution_rubber_bevel
            spline0.use_cyclic_u = True
            spline0.bezier_points.add(3)
            spline0.bezier_points[0].co = p0 + Vector((0.0, -s0, 0.0))
            spline0.bezier_points[0].handle_left_type = h
            spline0.bezier_points[0].handle_right_type = h
            spline0.bezier_points[1].co = p0 + Vector((-s0, 0.0, 0.0))
            spline0.bezier_points[1].handle_left_type = h
            spline0.bezier_points[1].handle_right_type = h
            spline0.bezier_points[2].co = p0 + Vector((0.0, s0, 0.0))
            spline0.bezier_points[2].handle_left_type = h
            spline0.bezier_points[2].handle_right_type = h
            spline0.bezier_points[3].co = p0 + Vector((s0, 0.0, 0.0))
            spline0.bezier_points[3].handle_left_type = h
            spline0.bezier_points[3].handle_right_type = h

        obj, cu1, spline1 = self.CreateCurve(name, layers, self.resolution_rubber)

        h = 'AUTO'
        p1 = Vector(self.geometry_correction((position_xy[0], position_xy[1], surface + 2.5)))
        s1 = (diameter[subtype] - 5.0) / 2.0
        spline1.use_cyclic_u = True
        spline1.resolution_u = self.resolution_rubber * 2
        spline1.bezier_points.add(3)
        spline1.bezier_points[0].co = p1 + Vector((0.0, -s1, 0.0))
        spline1.bezier_points[0].handle_left_type = h
        spline1.bezier_points[0].handle_right_type = h
        spline1.bezier_points[1].co = p1 + Vector((-s1, 0.0, 0.0))
        spline1.bezier_points[1].handle_left_type = h
        spline1.bezier_points[1].handle_right_type = h
        spline1.bezier_points[2].co = p1 + Vector((0.0, s1, 0.0))
        spline1.bezier_points[2].handle_left_type = h
        spline1.bezier_points[2].handle_right_type = h
        spline1.bezier_points[3].co = p1 + Vector((s1, 0.0, 0.0))
        spline1.bezier_points[3].handle_left_type = h
        spline1.bezier_points[3].handle_right_type = h

        cu1.bevel_object = wire_bevel

        return obj

    def CreateGuideWire(self, name, layers, fpx_points, surface, height, width):
        obj, cu, act_spline = self.CreateCurve(name, layers, self.resolution_wire)

        if height is None:
            height = 0.0
        if width is not None:
            bevel_name = "__fpx_guide_wire_bevel_{}__".format(width)
            wire_bevel = self.__blend_data.objects.get(bevel_name)
            if wire_bevel is None:
                if ops.curve.primitive_bezier_circle_add.poll():
                    ops.curve.primitive_bezier_circle_add()
                    wire_bevel = self.__context.active_object
                    wire_bevel.name = bevel_name
                    wire_bevel.data.dimensions = '2D'
                    wire_bevel.data.resolution_u = self.resolution_wire_bevel
                    wire_bevel.data.splines[0].resolution_u = self.resolution_wire_bevel
                    scale = width / 2.0
                    wire_bevel.scale = Vector((scale,scale,scale))
            cu.bevel_object = wire_bevel
            cu.use_fill_caps = True
        else:
            width = 0.0

        act_spline.use_cyclic_u = False
        self.CreateCurvePoints(act_spline, fpx_points, (surface + height + width / 2.0))

        # create pole caps
        co1 = act_spline.bezier_points[0].co
        h_left1 = act_spline.bezier_points[0].handle_left
        h_right1 = act_spline.bezier_points[0].handle_right
        co2 = act_spline.bezier_points[-1].co
        h_left2 = act_spline.bezier_points[-1].handle_left
        h_right2 = act_spline.bezier_points[-1].handle_right
        self.CreateWirePole(cu.splines, co1, h_left1, h_right1, surface, width)
        self.CreateWirePole(cu.splines, co2, h_right2, h_left2, surface, width)

        # merge wire curve with pole caps
        self.__context.scene.objects.active = obj
        self.MergeCaps(cu.splines, width)

        cu.splines[0].type = 'NURBS' # looks better for wires
        cu.twist_mode = 'MINIMUM'

        return obj

    def CreateGuideWall(self, name, layers, fpx_points, surface, height, width):
        obj, cu, act_spline = self.CreateCurve(name, layers, self.resolution_shape)

        modifier_solidify = obj.modifiers.new("width", type='SOLIDIFY')
        modifier_solidify.thickness = width
        modifier_solidify.offset = 0.0
        modifier_solidify.use_even_offset = True

        modifier_edge_split = obj.modifiers.new("edge_split", type='EDGE_SPLIT')

        if height is None:
            height = 0.0
        cu.extrude = height / 2.0
        cu.dimensions = '2D'

        act_spline.use_cyclic_u = False
        self.CreateCurvePoints(act_spline, fpx_points)

        obj.location = Vector((obj.location.x, obj.location.y, (surface + cu.extrude)))
        return obj

    def CreateLightRound(self, name, layers, position_xy, surface, diameter):
        obj, cu, spline = self.CreateCurve(name, layers, self.resolution_shape)

        modifier_edge_split = obj.modifiers.new("edge_split", type='EDGE_SPLIT')

        h = 'AUTO'
        p0 = Vector((0.0, 0.0, 0.0))
        d = diameter / 2.0
        spline.bezier_points.add(3)
        spline.bezier_points[0].co = p0 + Vector((0.0, -d, 0.0))
        spline.bezier_points[0].handle_left_type = h
        spline.bezier_points[0].handle_right_type = h
        spline.bezier_points[1].co = p0 + Vector((-d, 0.0, 0.0))
        spline.bezier_points[1].handle_left_type = h
        spline.bezier_points[1].handle_right_type = h
        spline.bezier_points[2].co = p0 + Vector((0.0, d, 0.0))
        spline.bezier_points[2].handle_left_type = h
        spline.bezier_points[2].handle_right_type = h
        spline.bezier_points[3].co = p0 + Vector((d, 0.0, 0.0))
        spline.bezier_points[3].handle_left_type = h
        spline.bezier_points[3].handle_right_type = h
        spline.use_cyclic_u = True

        cu.extrude = self.debug_light_extrude
        cu.dimensions = '2D'

        obj.location = Vector((obj.location.x, obj.location.y, surface))
        return obj

    def CreateLightShapeable(self, name, layers, fpx_points, surface):
        obj, cu, act_spline = self.CreateCurve(name, layers, self.resolution_shape)

        modifier_edge_split = obj.modifiers.new("edge_split", type='EDGE_SPLIT')

        cu.extrude = self.debug_light_extrude
        cu.dimensions = '2D'

        act_spline.use_cyclic_u = True
        self.CreateCurvePoints(act_spline, fpx_points)

        obj.location = Vector((obj.location.x, obj.location.y, surface))
        return obj

    def CreateLightImage(self, name, layers, position_xy, surface, height, width, rotation):
        mesh = self.__blend_data.meshes.new(FORMAT_MESH.format(name))
        obj = self.__blend_data.objects.new(FORMAT_MESH_OBJECT.format(name), mesh)
        self.__context.scene.objects.link(obj)

        z = surface + self.debug_light_extrude
        bm = bmesh.new()
        uv_layer = bm.loops.layers.uv.new("UVMap")

        bmv_list = []
        bmv = bm.verts.new(Vector(self.geometry_correction((-width / 2.0, height / 2.0, 0.0))))
        bmv_list.append(bmv)
        bmv = bm.verts.new(Vector(self.geometry_correction((width / 2.0, height / 2.0, 0.0))))
        bmv_list.append(bmv)
        bmv = bm.verts.new(Vector(self.geometry_correction((width / 2.0, -height / 2.0, 0.0))))
        bmv_list.append(bmv)
        bmv = bm.verts.new(Vector(self.geometry_correction((-width / 2.0, -height / 2.0, 0.0))))
        bmv_list.append(bmv)
        bmf = bm.faces.new(bmv_list)
        bmluv = bmf.loops[0][uv_layer]
        bmluv.uv = (0.0, 0.0)
        bmluv = bmf.loops[1][uv_layer]
        bmluv.uv = (1.0, 0.0)
        bmluv = bmf.loops[2][uv_layer]
        bmluv.uv = (1.0, 1.0)
        bmluv = bmf.loops[3][uv_layer]
        bmluv.uv = (0.0, 1.0)
        tex_layer = bm.faces.layers.tex.new()
        bm.to_mesh(mesh)
        bm.free()

        obj.location = Vector(self.geometry_correction((position_xy[0], position_xy[1], z)))
        obj.rotation_mode = 'XZY'
        obj.rotation_euler = Euler((0.0, 0.0, radians(self.angle_correction(rotation))), obj.rotation_mode)
        return obj

    def CreateWireRamp(self, name, layers, fpx_points, surface, start_height, end_height, fpx_id, model_name_start, model_name_end):
        if start_height is None:
            start_height = 0.0
        if end_height is None:
            end_height = 0.0

        wire_bevel = self.PrepareWireRampBevel()

        #"ramp_point"
        obj, cu, act_spline = self.CreateCurve(name, layers, self.resolution_wire)

        cu.bevel_object = wire_bevel
        cu.use_fill_caps = True

        act_spline.use_cyclic_u = False
        self.CreateRampCurvePoints(act_spline, fpx_points, surface, start_height, end_height)


        # ramp start
        if model_name_start:
            blender_empty_object = self.__blend_data.objects.new(FORMAT_EMPTY_OBJECT.format(FORMAT_MODEL_CAP1.format(name)), None)
            blender_empty_object.location = cu.splines[-1].bezier_points[0].co
            blender_empty_object.rotation_mode = 'XZY'
            v = (cu.splines[-1].bezier_points[0].handle_left - cu.splines[-1].bezier_points[0].co)
            blender_empty_object.rotation_euler = Euler((0, 0, Vector((v.x, v.y)).angle_signed(Vector((1.0, 0.0)))), 'XZY')
            blender_empty_object.empty_draw_type = 'ARROWS'
            blender_empty_object.empty_draw_size = 10.0
            self.__context.scene.objects.link(blender_empty_object)
            blender_empty_object.fpt.name = FORMAT_EMPTY_OBJECT.format(FORMAT_MODEL_START.format(name))
            if fpx_id:
                blender_empty_object.fpt.id = FptElementType.VALUE_INT_TO_NAME.get(fpx_id)
            self.attach_dupli_group(blender_empty_object, FptImporter.LAYERS_WIRE_RING, model_name_start, "model_start")
            blender_empty_object.layers = FptImporter.LAYERS_WIRE_RING

        # ramp end
        if model_name_end:
            blender_empty_object = self.__blend_data.objects.new(FORMAT_EMPTY_OBJECT.format(FORMAT_MODEL_CAP2.format(name)), None)
            blender_empty_object.location = cu.splines[-1].bezier_points[-1].co
            blender_empty_object.rotation_mode = 'XZY'
            v = (cu.splines[-1].bezier_points[-1].handle_right - cu.splines[-1].bezier_points[-1].co)
            blender_empty_object.rotation_euler = Euler((0, 0, Vector((v.x, v.y)).angle_signed(Vector((1.0, 0.0)))), 'XZY')
            blender_empty_object.empty_draw_type = 'ARROWS'
            blender_empty_object.empty_draw_size = 10.0
            self.__context.scene.objects.link(blender_empty_object)
            blender_empty_object.fpt.name = FORMAT_EMPTY_OBJECT.format(FORMAT_MODEL_END.format(name))
            if fpx_id:
                blender_empty_object.fpt.id = FptElementType.VALUE_INT_TO_NAME.get(fpx_id)
            self.attach_dupli_group(blender_empty_object, FptImporter.LAYERS_WIRE_RING, model_name_end, "model_end")
            blender_empty_object.layers = FptImporter.LAYERS_WIRE_RING

        # create rings
        wire_ring_model = [
                None, # NoRing
                'wirering01', # FullRing = 1
                'wirering02', # OpenUp = 2
                'wirering04', # OpenRight = 3
                'wirering03', # OpenDown = 4
                'wirering05', # OpenLeft = 5
                'wirering06', # HalfLeft = 6
                'wirering07', # HalfRight = 7
                'wirering08', # Joiner = 8
                'wirering09', # HalfDown = 9
                'wirering10', # HalfUp = 10
                'wirering14', # OpenUPLeft = 11
                'wirering13', # OpenDownLeft = 12
                'wirering12', # OpenDownRight = 13
                'wirering11', # OpenUPRight = 14
                'wirering15', # Split = 15
                'wirering17', # QuarterRight = 16
                'wirering16', # QuarterLeft = 17
                'wirering19', # CresentRight = 18
                'wirering18', # CresentLeft = 19
                ]

        left_wires_bevel = self.PrepareWireRampSideBevel(2)
        right_wires_bevel = self.PrepareWireRampSideBevel(3)
        left_upper_wires_bevel = self.PrepareWireRampSideBevel(4)
        right_upper_wires_bevel = self.PrepareWireRampSideBevel(5)
        top_wires_bevel = self.PrepareWireRampSideBevel(6)

        last_bezier_point = None
        last_fpx_point = None

        last_left_wire = None
        last_right_wire = None
        last_left_upper_wire = None
        last_right_upper_wire = None
        last_top_wire = None

        for index, fpx_point in enumerate(fpx_points):
            bezier_point = act_spline.bezier_points[index]

            if self.debug_create_full_ramp_wires:
                pass
            else:
                """
                there are problems, see [#36007] http://projects.blender.org/tracker/index.php?func=detail&aid=36007&group_id=9&atid=498
                """
                if index:
                    if last_fpx_point.get_value("left_guide"):
                        last_left_wire = self.CreateWireRampGuidePiece(name, obj, layers, left_wires_bevel, 2, index, last_bezier_point, bezier_point, last_left_wire)
                    else:
                        last_left_wire = None

                    if last_fpx_point.get_value("right_guide"):
                        last_right_wire = self.CreateWireRampGuidePiece(name, obj, layers, right_wires_bevel, 3, index, last_bezier_point, bezier_point, last_right_wire)
                    else:
                        last_right_wire = None

                    if last_fpx_point.get_value("left_upper_guide"):
                        last_left_upper_wire = self.CreateWireRampGuidePiece(name, obj, layers, left_upper_wires_bevel, 4, index, last_bezier_point, bezier_point, last_left_upper_wire)
                    else:
                        last_left_upper_wire = None

                    if last_fpx_point.get_value("right_upper_guide"):
                        last_right_upper_wire = self.CreateWireRampGuidePiece(name, obj, layers, right_upper_wires_bevel, 5, index, last_bezier_point, bezier_point, last_right_upper_wire)
                    else:
                        last_right_upper_wire = None

                    if last_fpx_point.get_value("top_wire"):
                        last_top_wire = self.CreateWireRampGuidePiece(name, obj, layers, top_wires_bevel, 6, index, last_bezier_point, bezier_point, last_top_wire)
                    else:
                        last_top_wire = None


            last_bezier_point = bezier_point
            last_fpx_point = fpx_point

            #fpx_point.get_value("mark_as_ramp_end_point")

            type = fpx_point.get_value("ring_type")
            raw_model_name = wire_ring_model[type]
            if raw_model_name is None:
                continue

            blender_empty_object = self.__blend_data.objects.new(FORMAT_EMPTY_OBJECT.format(FORMAT_MODEL_RING.format(name, index)), None)
            blender_empty_object.location = bezier_point.co
            blender_empty_object.rotation_mode = 'XZY'
            v = (bezier_point.handle_right - bezier_point.co)
            blender_empty_object.rotation_euler = Euler((0, 0, Vector((v.x, v.y)).angle_signed(Vector((1.0, 0.0)))), 'XZY')
            blender_empty_object.empty_draw_type = 'ARROWS'
            blender_empty_object.empty_draw_size = 10.0
            self.__context.scene.objects.link(blender_empty_object)
            blender_empty_object.fpt.name = FORMAT_MODEL_RING.format(name, index)
            if fpx_id:
                blender_empty_object.fpt.id = FptElementType.VALUE_INT_TO_NAME.get(fpx_id)
            self.attach_dupli_group(blender_empty_object, FptImporter.LAYERS_WIRE_RING, raw_model_name, 'LOCAL')
            blender_empty_object.layers = FptImporter.LAYERS_WIRE_RING

        #cu.splines[0].type = 'NURBS' # looks better for wires
        #cu.twist_mode = 'MINIMUM'
        return obj

    def CreateWireRampGuidePiece(self, name, parent_obj, layers, wire_bevel, wire_index, point_index, last_bezier_point_template, bezier_point_template, last_object):
        if last_object:
            #reuse previouse curve
            spline = last_object.data.splines[0]
            spline.bezier_points.add(1)
            bezier_point = spline.bezier_points[-1]
            bezier_point.co = bezier_point_template.co
            bezier_point.radius = bezier_point_template.radius
            bezier_point.handle_left_type = bezier_point_template.handle_left_type
            bezier_point.handle_right_type = bezier_point_template.handle_right_type
            bezier_point.handle_left = bezier_point_template.handle_left
            bezier_point.handle_right = bezier_point_template.handle_right
            bezier_point.tilt = bezier_point_template.tilt
            obj = last_object
        else:
            #start to make a new curve
            sub_name = "{}_{}_{}".format(name, wire_index, point_index-1)

            obj, cu, spline = self.CreateCurve(sub_name, layers, self.resolution_wire)
            obj.fpt.name = sub_name
            obj.parent = parent_obj
            cu.bevel_object = wire_bevel
            cu.use_fill_caps = True
            spline.use_cyclic_u = False

            spline.bezier_points.add(1)
            bezier_point = spline.bezier_points[0]
            bezier_point.co = last_bezier_point_template.co
            bezier_point.radius = last_bezier_point_template.radius
            bezier_point.handle_left_type = last_bezier_point_template.handle_left_type
            bezier_point.handle_right_type = last_bezier_point_template.handle_right_type
            bezier_point.handle_left = last_bezier_point_template.handle_left
            bezier_point.handle_right = last_bezier_point_template.handle_right
            bezier_point.tilt = last_bezier_point_template.tilt

            bezier_point = spline.bezier_points[1]
            bezier_point.co = bezier_point_template.co
            bezier_point.radius = bezier_point_template.radius
            bezier_point.handle_left_type = bezier_point_template.handle_left_type
            bezier_point.handle_right_type = bezier_point_template.handle_right_type
            bezier_point.handle_left = bezier_point_template.handle_left
            bezier_point.handle_right = bezier_point_template.handle_right
            bezier_point.tilt = bezier_point_template.tilt

        return obj

    def CreateRamp(self, name, layers, fpx_points, surface, start_height, end_height, start_width, end_width, height_left, height_right):
        if start_width is None:
            start_width = 0.0
        if end_width is None:
            end_width = 0.0

        if height_left is None:
            height_left = 0.0
        if height_right is None:
            height_right = 0.0

        bevel_name = "__fpx_guide_ramp_wire_bevel_{}_{}_{}__".format(start_width, height_left, height_right, )
        wire_bevel = self.__blend_data.objects.get(bevel_name)
        if wire_bevel is None:
            cu = self.__blend_data.curves.new(bevel_name, 'CURVE')
            wire_bevel = self.__blend_data.objects.new(bevel_name, cu)
            self.__context.scene.objects.link(wire_bevel)
            cu.dimensions = '2D'
            cu.resolution_u = self.resolution_shape

            h = 'VECTOR'
            cu.splines.new('BEZIER')
            spline0 = cu.splines[-1]
            spline0.resolution_u = self.resolution_shape
            p0 = Vector((0.0, 0.0, 0.0))
            spline0.use_cyclic_u = False
            spline0.bezier_points.add(3)
            spline0.bezier_points[0].co = p0 + Vector((-start_width / 2.0, height_left, 0.0))
            spline0.bezier_points[0].handle_left_type = h
            spline0.bezier_points[0].handle_right_type = h
            spline0.bezier_points[1].co = p0 + Vector((-start_width / 2.0, 0.0, 0.0))
            spline0.bezier_points[1].handle_left_type = h
            spline0.bezier_points[1].handle_right_type = h
            spline0.bezier_points[2].co = p0 + Vector((start_width / 2.0, 0.0, 0.0))
            spline0.bezier_points[2].handle_left_type = h
            spline0.bezier_points[2].handle_right_type = h
            spline0.bezier_points[3].co = p0 + Vector((start_width / 2.0, height_right, 0.0))
            spline0.bezier_points[3].handle_left_type = h
            spline0.bezier_points[3].handle_right_type = h

        #"ramp_point"
        obj, cu, act_spline = self.CreateCurve(name, layers, self.resolution_wire)

        modifier_solidify = obj.modifiers.new("solidify", type='SOLIDIFY')
        modifier_solidify.offset = 0.0
        modifier_solidify.thickness = 1.0
        modifier_edge_split = obj.modifiers.new("edge_split", type='EDGE_SPLIT')

        cu.bevel_object = wire_bevel
        cu.use_fill_caps = False


        if start_height is None:
            start_height = 0.0
        if end_height is None:
            end_height = 0.0

        self.CreateRampCurvePoints(act_spline, fpx_points, surface, start_height, end_height, start_width, end_width)

        return obj

    def CreateWirePole(self, cu_splines, co, t, ti, surface, width):
        d = (t - co)
        dn = d.normalized()
        w = width / 2.0
        dw = dn * w
        co_dw = co + dw

        cu_splines.new('BEZIER')
        act_spline = cu_splines[-1]
        act_spline.use_cyclic_u = False

        point_0 = act_spline.bezier_points[-1]
        point_0.co = Vector((co_dw.x, co_dw.y, co.z - w))
        point_0.handle_left_type = 'ALIGNED'
        point_0.handle_right_type = 'ALIGNED'
        point_0.handle_left = Vector((point_0.co.x, point_0.co.y, point_0.co.z + w))
        point_0.handle_right = Vector((point_0.co.x, point_0.co.y, point_0.co.z - w))

        act_spline.bezier_points.add()
        point_1 = act_spline.bezier_points[-1]
        point_1.co = Vector((co_dw.x, co_dw.y, surface))
        point_1.handle_left_type = 'ALIGNED'
        point_1.handle_right_type = 'ALIGNED'
        point_1.handle_left = Vector((point_1.co.x, point_1.co.y, point_1.co.z + (co.z - surface) / 4.0))
        point_1.handle_right = Vector((point_1.co.x, point_1.co.y, point_1.co.z - (co.z - surface) / 4.0))

    def MergeCaps(self, cu_splines, width):
        w = width / 2.0

        # adjust endpoint of curve
        b_point = cu_splines[0].bezier_points[0]
        co = b_point.co.copy()
        h_left = b_point.handle_left.copy()
        h_right = b_point.handle_right.copy()
        b_point.handle_left_type = 'ALIGNED'
        b_point.handle_right_type = 'ALIGNED'
        b_point.handle_left = co + ((h_left - co).normalized() * w)
        b_point.handle_right = co + ((h_right - co).normalized() * w)
        # select endpoint of curve and start-point of pole-cap
        cu_splines[0].bezier_points[0].select_control_point = True
        cu_splines[1].bezier_points[0].select_control_point = True
        # merge curve an pole
        FpxUtilities.enable_edit_mode(True, self.__context)
        if ops.curve.make_segment.poll():
            ops.curve.make_segment()
        FpxUtilities.enable_edit_mode(False, self.__context)

        # adjust endpoint of curve
        b_point = cu_splines[0].bezier_points[-1]
        co = b_point.co.copy()
        h_left = b_point.handle_left.copy()
        h_right = b_point.handle_right.copy()
        b_point.handle_left_type = 'ALIGNED'
        b_point.handle_right_type = 'ALIGNED'
        b_point.handle_left = co + ((h_left - co).normalized() * w)
        b_point.handle_right = co + ((h_right - co).normalized() * w)
        # select endpoint of curve and start-point of pole-cap
        cu_splines[0].bezier_points[-1].select_control_point = True
        cu_splines[1].bezier_points[0].select_control_point = True
        # merge curve an pole
        FpxUtilities.enable_edit_mode(True, self.__context)
        if ops.curve.make_segment.poll():
            ops.curve.make_segment()
        FpxUtilities.enable_edit_mode(False, self.__context)

    def CreateCurve(self, name, layers, curve_resolution):
        cu = self.__blend_data.curves.new(FORMAT_CURVE.format(name), 'CURVE')
        obj = self.__blend_data.objects.new(FORMAT_CURVE_OBJECT.format(name), cu)
        self.__context.scene.objects.link(obj)

        cu.dimensions = '3D'
        cu.twist_mode = 'Z_UP'
        cu.splines.new('BEZIER')
        spline = cu.splines[-1]
        spline.resolution_u = curve_resolution
        cu.resolution_u = curve_resolution

        obj.layers = layers
        return obj, cu, spline

    def CreateRampCurvePoints(self, spline, fpx_points, z, z0, z1, w0=1.0, w1=1.0):
        ramp_length_sum = 0.0
        ramp_length = []
        last_point = None
        for fpx_point in fpx_points:
            fpx_position_xy = Vector(fpx_point.get_value("position"))
            if last_point:
                length = (fpx_position_xy - last_point).length
                ramp_length_sum += length
            ramp_length.append(ramp_length_sum)
            last_point = fpx_position_xy

        # CreateCurvePoints & radius
        spline.bezier_points.add(len(fpx_points) - 1)

        for index, fpx_point in enumerate(fpx_points):
            fpx_position_xy = fpx_point.get_value("position")
            fpx_smooth = fpx_point.get_value("smooth")

            factor = (ramp_length[index] / ramp_length_sum)
            offset = (z1 - z0) * factor

            bezier_point = spline.bezier_points[index]
            bezier_point.co = Vector(self.geometry_correction((fpx_position_xy[0], fpx_position_xy[1], (z + z0 + offset))))
            bezier_point.radius = (w0 + ((w1 - w0) * factor)) / w0

            if fpx_smooth:
                handle_type = 'AUTO'
            else:
                handle_type = 'VECTOR'

            bezier_point.handle_left_type = handle_type
            bezier_point.handle_right_type = handle_type

        if self.use_hermite_handle:
            self.SetCatmullRomHermiteBezierHandle(spline)

    def CreateCurvePoints(self, spline, fpx_points, z=0.0):
        spline.bezier_points.add(len(fpx_points) - 1)

        for index, fpx_point in enumerate(fpx_points):
            fpx_position_xy = fpx_point.get_value("position")
            fpx_smooth = fpx_point.get_value("smooth")

            bezier_point = spline.bezier_points[index]
            bezier_point.co = Vector(self.geometry_correction((fpx_position_xy[0], fpx_position_xy[1], z)))

            if fpx_smooth:
                handle_type = 'AUTO'
            else:
                handle_type = 'VECTOR'

            bezier_point.handle_left_type = handle_type
            bezier_point.handle_right_type = handle_type

        if self.use_hermite_handle:
            self.SetCatmullRomHermiteBezierHandle(spline)

    def SetCatmullRomHermiteBezierHandle(self, spline):
        if spline.type != 'BEZIER':
            return
        count_bezier_point = len(spline.bezier_points)
        max_index_bezier_point = count_bezier_point - 1
        min_index_bezier_point = 0

        ## Catmull-Rom
        catmull_rom_vectors = []
        for index_bezier_point in range(count_bezier_point):
            if index_bezier_point > 0 and index_bezier_point < max_index_bezier_point:
                point_prev = spline.bezier_points[index_bezier_point - 1].co
                point_next = spline.bezier_points[index_bezier_point + 1].co
                catmull_rom_vector = (point_next - point_prev) / 2.0
            elif not spline.use_cyclic_u and index_bezier_point == 0:
                point_prev = spline.bezier_points[index_bezier_point].co
                point_next = spline.bezier_points[index_bezier_point + 1].co
                catmull_rom_vector = (point_next - point_prev) / 2.0
            elif not spline.use_cyclic_u and index_bezier_point == max_index_bezier_point:
                point_prev = spline.bezier_points[index_bezier_point - 1].co
                point_next = spline.bezier_points[index_bezier_point].co
                catmull_rom_vector = (point_next - point_prev) / 2.0
            elif spline.use_cyclic_u and index_bezier_point == 0:
                point_prev = spline.bezier_points[max_index_bezier_point].co
                point_next = spline.bezier_points[index_bezier_point + 1].co
                catmull_rom_vector = (point_next - point_prev) / 2.0
            elif spline.use_cyclic_u and index_bezier_point == max_index_bezier_point:
                point_prev = spline.bezier_points[index_bezier_point - 1].co
                point_next = spline.bezier_points[min_index_bezier_point].co
                catmull_rom_vector = (point_next - point_prev) / 2.0
            else:
                catmull_rom_vector = Vector()
            catmull_rom_vectors.append(catmull_rom_vector)

        ## Hermite to Cubic Bezier right handles
        for index_bezier_point in range(count_bezier_point):
            bezier_point = spline.bezier_points[index_bezier_point]
            point = bezier_point.co
            if bezier_point.handle_right_type in {'VECTOR', }:
                if index_bezier_point < max_index_bezier_point:
                    bezier_point_next = spline.bezier_points[index_bezier_point + 1]
                    point_next = bezier_point_next.co
                    catmull_rom_vector = (point_next - point) / 2.0
                elif spline.use_cyclic_u and index_bezier_point == max_index_bezier_point:
                    bezier_point_next = spline.bezier_points[min_index_bezier_point]
                    point_next = bezier_point_next.co
                    catmull_rom_vector = (point_next - point) / 2.0
                else:
                    bezier_point_prev = spline.bezier_points[index_bezier_point - 1]
                    point_prev = bezier_point_prev.co
                    catmull_rom_vector = (point - point_prev) / 2.0
            else:
                catmull_rom_vector = catmull_rom_vectors[index_bezier_point]
            bezier_handle_point = point + (catmull_rom_vector / 3.0)
            bezier_point.handle_right_type = 'FREE'
            bezier_point.handle_right = bezier_handle_point

        ## Hermite to Cubic Bezier left handles
        for index_bezier_point in range(count_bezier_point):
            bezier_point = spline.bezier_points[index_bezier_point]
            point = bezier_point.co
            if bezier_point.handle_left_type in {'VECTOR', }:
                bezier_point.handle_left_type = 'FREE'
                if index_bezier_point > 0:
                    bezier_point_prev = spline.bezier_points[index_bezier_point - 1]
                    point_prev = bezier_point_prev.co
                    catmull_rom_vector = (point - point_prev) / 2.0
                elif spline.use_cyclic_u and index_bezier_point == min_index_bezier_point:
                    bezier_point_prev = spline.bezier_points[max_index_bezier_point]
                    point_prev = bezier_point_prev.co
                    catmull_rom_vector = (point - point_prev) / 2.0
                else:
                    bezier_point_next = spline.bezier_points[index_bezier_point + 1]
                    point_next = bezier_point_next.co
                    catmull_rom_vector = (point_next - point) / 2.0
            else:
                bezier_point.handle_left_type = 'ALIGNED'
                bezier_point.handle_right_type = 'ALIGNED'
                catmull_rom_vector = catmull_rom_vectors[index_bezier_point]
            bezier_handle_point = point - (catmull_rom_vector / 3.0)
            #bezier_point.handle_left_type = 'FREE'
            bezier_point.handle_left = bezier_handle_point

    def CreatePinCab(self, fpx_table_data):
        name = fpx_table_data.get_value("name")
        name = FpxUtilities.toGoodName(name) ####

        dx = fpx_table_data.get_value("length")
        dy = fpx_table_data.get_value("width")
        z0 = fpx_table_data.get_value("glass_height_front")
        z1 = fpx_table_data.get_value("glass_height_rear")

        mesh = self.__blend_data.meshes.new(FORMAT_MESH.format(name))
        obj = self.__blend_data.objects.new(FORMAT_MESH_OBJECT.format(name), mesh)
        self.__context.scene.objects.link(obj)

        bm = bmesh.new()
        uv_layer = bm.loops.layers.uv.new("UVMap")

        #inner playfield
        bmv_list = []
        bmv = bm.verts.new(Vector((dx, 0.0, 0.0)))
        bmv_list.append(bmv)
        bmv = bm.verts.new(Vector((dx, dy, 0.0)))
        bmv_list.append(bmv)
        bmv = bm.verts.new(Vector((0.0, dy, 0.0)))
        bmv_list.append(bmv)
        bmv = bm.verts.new(Vector((0.0, 0.0, 0.0)))
        bmv_list.append(bmv)
        bmf = bm.faces.new(bmv_list)
        bmluv = bmf.loops[0][uv_layer]
        bmluv.uv = (0.0, 0.0)
        bmluv = bmf.loops[1][uv_layer]
        bmluv.uv = (1.0, 0.0)
        bmluv = bmf.loops[2][uv_layer]
        bmluv.uv = (1.0, 1.0)
        bmluv = bmf.loops[3][uv_layer]
        bmluv.uv = (0.0, 1.0)
        tex_layer = bm.faces.layers.tex.new()
        bm.to_mesh(mesh)
        bm.free()

        mesh_box = self.__blend_data.meshes.new(FORMAT_MESH.format("{}.playfield".format(name)))
        obj_box = self.__blend_data.objects.new(FORMAT_MESH_OBJECT.format("{}.playfield".format(name)), mesh_box)
        obj_box.parent = obj
        self.__context.scene.objects.link(obj_box)

        bm = bmesh.new()
        #inner back
        bmv_list = []
        bmv = bm.verts.new(Vector((0.0, 0.0, 0.0)))
        bmv_list.append(bmv)
        bmv = bm.verts.new(Vector((0.0, dy, 0.0)))
        bmv_list.append(bmv)
        bmv = bm.verts.new(Vector((0.0, dy, z1)))
        bmv_list.append(bmv)
        bmv = bm.verts.new(Vector((0.0, 0.0, z1)))
        bmv_list.append(bmv)
        bmf = bm.faces.new(bmv_list)

        #inner front
        bmv_list = []
        bmv = bm.verts.new(Vector((dx, 0.0, 0.0)))
        bmv_list.append(bmv)
        bmv = bm.verts.new(Vector((dx, 0.0, z0)))
        bmv_list.append(bmv)
        bmv = bm.verts.new(Vector((dx, dy, z0)))
        bmv_list.append(bmv)
        bmv = bm.verts.new(Vector((dx, dy, 0.0)))
        bmv_list.append(bmv)
        bmf = bm.faces.new(bmv_list)

        #inner left
        bmv_list = []
        bmv = bm.verts.new(Vector((0.0, 0.0, 0.0)))
        bmv_list.append(bmv)
        bmv = bm.verts.new(Vector((0.0, 0.0, z1)))
        bmv_list.append(bmv)
        bmv = bm.verts.new(Vector((dx, 0.0, z0)))
        bmv_list.append(bmv)
        bmv = bm.verts.new(Vector((dx, 0.0, 0.0)))
        bmv_list.append(bmv)
        bmf = bm.faces.new(bmv_list)

        #inner right
        bmv_list = []
        bmv = bm.verts.new(Vector((0.0, dy, 0.0)))
        bmv_list.append(bmv)
        bmv = bm.verts.new(Vector((dx, dy, 0.0)))
        bmv_list.append(bmv)
        bmv = bm.verts.new(Vector((dx, dy, z0)))
        bmv_list.append(bmv)
        bmv = bm.verts.new(Vector((0.0, dy, z1)))
        bmv_list.append(bmv)
        bmf = bm.faces.new(bmv_list)

        bm.to_mesh(mesh_box)
        bm.free()

        ##
        dty = fpx_table_data.get_value("translite_height")
        if not dty:
            dty = 676.0
        dtz = fpx_table_data.get_value("translite_width")
        if not dtz:
            dtz = 676.0

        mesh_translite = self.__blend_data.meshes.new(FORMAT_MESH.format("{}.translite".format(name)))
        obj_translite = self.__blend_data.objects.new(FORMAT_MESH_OBJECT.format("{}.translite".format(name)), mesh_translite)
        obj_translite.parent = obj
        self.__context.scene.objects.link(obj_translite)

        bm = bmesh.new()
        uv_layer = bm.loops.layers.uv.new("UVMap")

        #inner translite
        bmv_list = []
        bmv = bm.verts.new(Vector((0.0, (dy - dty) / 2.0, z1 + 20.0)))
        bmv_list.append(bmv)
        bmv = bm.verts.new(Vector((0.0, (dy + dty) / 2.0, z1 + 20.0)))
        bmv_list.append(bmv)
        bmv = bm.verts.new(Vector((0.0, (dy + dty) / 2.0, z1 + 20.0 + dtz)))
        bmv_list.append(bmv)
        bmv = bm.verts.new(Vector((0.0, (dy - dty) / 2.0, z1 + 20.0 + dtz)))
        bmv_list.append(bmv)
        bmf = bm.faces.new(bmv_list)
        bmluv = bmf.loops[0][uv_layer]
        bmluv.uv = (0.0, 0.0)
        bmluv = bmf.loops[1][uv_layer]
        bmluv.uv = (1.0, 0.0)
        bmluv = bmf.loops[2][uv_layer]
        bmluv.uv = (1.0, 1.0)
        bmluv = bmf.loops[3][uv_layer]
        bmluv.uv = (0.0, 1.0)
        tex_layer = bm.faces.layers.tex.new()
        bm.to_mesh(mesh_translite)
        bm.free()

        return obj

    def CreateWireRampBevel(self, curve, wire_index):
        wire_diameter = [Vector((0.0, -2.0, 0.0)), Vector((-2.0, 0.0, 0.0)), Vector((0.0, 2.0, 0.0)), Vector((2.0, 0.0, 0.0))]
        wire_position = [Vector((-11.0, -2.0, 0.0)), Vector((11.0, -2.0, 0.0)), Vector((-17.0, 11, 0.0)), Vector((17.0, 11.0, 0.0)), Vector((-11.0, 24.0, 0.0)), Vector((11.0, 24.0, 0.0)), Vector((0.0, 33.0, 0.0))]
        w0 = Vector((0.0, 0.0, 0.0))
        handle = 'AUTO'
        curve.splines.new('BEZIER')
        p0 = wire_position[wire_index] + w0
        spline = curve.splines[-1]
        spline.resolution_u = self.resolution_wire_bevel
        spline.use_cyclic_u = True
        spline.bezier_points.add(3)
        spline.bezier_points[0].co = p0 + wire_diameter[0]
        spline.bezier_points[0].handle_left_type = handle
        spline.bezier_points[0].handle_right_type = handle
        spline.bezier_points[1].co = p0 + wire_diameter[1]
        spline.bezier_points[1].handle_left_type = handle
        spline.bezier_points[1].handle_right_type = handle
        spline.bezier_points[2].co = p0 + wire_diameter[2]
        spline.bezier_points[2].handle_left_type = handle
        spline.bezier_points[2].handle_right_type = handle
        spline.bezier_points[3].co = p0 + wire_diameter[3]
        spline.bezier_points[3].handle_left_type = handle
        spline.bezier_points[3].handle_right_type = handle

    def PrepareWireRampBevel(self):
        bevel_name = "__fpx_guide_ramp_wire_bevel__"
        wire_bevel = self.__blend_data.objects.get(bevel_name)
        if wire_bevel is None:
            cu = self.__blend_data.curves.new(bevel_name, 'CURVE')
            wire_bevel = self.__blend_data.objects.new(bevel_name, cu)
            self.__context.scene.objects.link(wire_bevel)
            cu.dimensions = '2D'
            cu.resolution_u = self.resolution_wire_bevel

            self.CreateWireRampBevel(cu, 0) # base left inner
            self.CreateWireRampBevel(cu, 1) # base right inner

            if self.debug_create_full_ramp_wires:
                """
                there are problems, see [#36007] http://projects.blender.org/tracker/index.php?func=detail&aid=36007&group_id=9&atid=498
                """
                self.CreateWireRampBevel(cu, 2) # left inner
                self.CreateWireRampBevel(cu, 3) # right inner
                self.CreateWireRampBevel(cu, 4) # upper left inner
                self.CreateWireRampBevel(cu, 5) # upper right inner
                self.CreateWireRampBevel(cu, 6) # top outer
            else:
                pass

        return wire_bevel

    def PrepareWireRampSideBevel(self, wire_index):
        bevel_name = "__fpx_guide_ramp_wire_bevel_{}__".format(wire_index)
        wire_bevel = self.__blend_data.objects.get(bevel_name)
        if wire_bevel is None:
            cu = self.__blend_data.curves.new(bevel_name, 'CURVE')
            wire_bevel = self.__blend_data.objects.new(bevel_name, cu)
            self.__context.scene.objects.link(wire_bevel)
            cu.dimensions = '2D'
            cu.resolution_u = self.resolution_wire_bevel

            self.CreateWireRampBevel(cu, wire_index)

        return wire_bevel

    ###########################################################################
    def geometry_correction(self, value):
        return Vector((value[1], value[0], value[2]))

    def angle_correction(self, value):
        return -value

    def attach_dupli_group(self, blender_empty_object, layers, fpx_model_name, fpx_type_name, offset=Vector(), angle=0):
        fpx_model_name = FpxUtilities.toGoodName(fpx_model_name) ####
        if fpx_model_name in self.debug_missing_resources:
            return

        if fpx_type_name == 'RAW':
            blender_group_name = FpxUtilities.toGoodName(FORMAT_GROUP.format(fpx_model_name))
            self.LoadObjectLite(blender_group_name, Fpt_PackedLibrary_Type.TYPE_MODEL)
        if fpx_type_name == 'LOCAL':
            blender_group_name = FpxUtilities.toGoodName(FORMAT_RESOURCE.format(PREFIX_LOCAL, FORMAT_GROUP.format(fpx_model_name)))
            self.LoadObjectLite(blender_group_name, Fpt_PackedLibrary_Type.TYPE_MODEL)
        else:
            fpx_model_object = self.fpx_pinmodels.get(fpx_model_name)
            if not fpx_model_object:
                if self.verbose in FpxUI.VERBOSE_NORMAL:
                    print("#DEBUG attach_dupli_group, fpx_pinmodel not found!", fpx_model_name, fpx_type_name)
                    self.debug_missing_resources.add(fpx_model_name)
                return
            blender_group_name = fpx_model_object[self.BLENDER_OBJECT_NAME]

        if self.__blend_data.groups.get(blender_group_name):
            blender_empty_object_new = self.__blend_data.objects.new(FORMAT_DUPLI_OBJECT.format(blender_empty_object.name), None)
            blender_empty_object_new.location = offset
            blender_empty_object_new.rotation_mode = blender_empty_object.rotation_mode
            blender_empty_object_new.rotation_euler = Euler((0, 0, radians(angle)), blender_empty_object.rotation_mode)
            blender_empty_object_new.empty_draw_type = blender_empty_object.empty_draw_type
            blender_empty_object_new.empty_draw_size = blender_empty_object.empty_draw_size
            self.__context.scene.objects.link(blender_empty_object_new)

            blender_empty_object_new.parent = blender_empty_object

            blender_empty_object_new.dupli_type = 'GROUP'
            blender_empty_object_new.dupli_group = self.__blend_data.groups[blender_group_name]

            blender_empty_object_new.layers = layers
        else:
            print("#DEBUG attach_dupli_group, blender_group not found!", fpx_model_name, fpx_type_name, blender_group_name)
            self.debug_missing_resources.add(fpx_model_name)

        # add name to fpt property
        blender_empty_object.fpt.add_model(fpx_type_name, fpx_model_name)

    ###########################################################################
    def FpxLayerToBlenderLayers(self, layer, id=None, render=None, alpha=None, sphere_mapping=None, crystal=None, base=None):
        if layer is None:
            layer = 0x000

        if id is None:
            id = 0

        if render is None:
            render = True

        if alpha is None:
            alpha = 10

        if crystal is None:
            crystal = False

        if base is None:
            base = False

        layers = []
        for index in range(20):
            # layers, left block, top row
            if index == 0:
                layers.append(True)
            elif index == 1 and (render and alpha > 0): # visible objects
                layers.append(True)
            elif index == 2 and (not render or alpha <= 0): # invisible objects
                layers.append(True)
            elif index == 3 and id in FptElementType.SET_CURVE_OBJECTS: # curve object
                layers.append(True)
            elif index == 4 and id in FptElementType.SET_MESH_OBJECTS: # mesh object
                layers.append(True)

            # layers, left block, bottom row
            elif index == 10 and id in FptElementType.SET_LIGHT_OBJECTS: # light object
                layers.append(True)
            elif index == 11 and id in FptElementType.SET_RUBBER_OBJECTS: # rubber object
                layers.append(True)
            elif index == 12 and ((id in FptElementType.SET_SPHERE_MAPPING_OBJECTS and sphere_mapping) or id in FptElementType.SET_WIRE_OBJECTS): # chrome object
                layers.append(True)
            elif index == 13 and (crystal or (alpha > 0 and alpha < 10)): # crystal and transparent but visible objects
                layers.append(True)
            elif index == 14: # TODO: base mesh object
                layers.append(False)

            # layers, right block, top row
            elif index == 5 and (layer & 0x002) == 0x002: # layer 2
                layers.append(True)
            elif index == 6 and (layer & 0x008) == 0x008: # layer 4
                layers.append(True)
            elif index == 7 and (layer & 0x020) == 0x020: # layer 6
                layers.append(True)
            elif index == 8 and (layer & 0x080) == 0x080: # layer 8
                layers.append(True)
            elif index == 9 and (layer & 0x200) == 0x200: # layer 0
                layers.append(True)

            # layers, right block, bottom row
            elif index == 15 and (layer & 0x001) == 0x001: # layer 1
                layers.append(True)
            elif index == 16 and (layer & 0x004) == 0x004: # layer 3
                layers.append(True)
            elif index == 17 and (layer & 0x010) == 0x010: # layer 5
                layers.append(True)
            elif index == 18 and (layer & 0x040) == 0x040: # layer 7
                layers.append(True)
            elif index == 19 and (layer & 0x100) == 0x100: # layer 9
                layers.append(True)

            else:
                layers.append(False)
        return tuple(layers)

    def LoadObjectLite(self, name, type):
        """
        name: 'fpmodel.fpl\\<object>.g', 'fpmodel.fpl\\<object>.i'
        type: 'IMAGE', 'MODEL'
        """
        obj = self.LoadFromEmbedded(name, type)

        if not obj:
            obj = self.LoadFromBlendLibrary(name, type, self.blend_resource_file)

        if not obj:
            print("#DEBUG resource finally not found", name, type, lib)

        return obj

    def LoadObject(self, name, type, lib):
        """
        name: 'fpmodel.fpl\\<object>.g', 'fpmodel.fpl\\<object>.i'
        type: 'IMAGE', 'MODEL'
        lib_name: fpmodel.fpl
        path_name: '.\\', 'C:\\FuturePinball\\Libraries\\', '<addons>\\io_scene_fpx\\'
        """
        #print("#DEBUG LoadObject", name, type, lib)

        obj = self.LoadFromEmbedded(name, type)

        if not obj and lib:
            obj = self.LoadFromPathLibrary(name, type, lib, self.folder_name)

        if not obj:
            obj = self.LoadFromBlendLibrary(name, type, self.blend_resource_file)

        if not obj and lib:
            obj = self.LoadFromPathLibrary(name, type, lib, self.path_libraries)

        if not obj:
            print("#DEBUG resource finally not found", name, type, lib)

        return obj

    def LoadFromEmbedded(self, name, type):
        #print("#DEBUG LoadFromEmbedded", name, type)

        if type == Fpt_PackedLibrary_Type.TYPE_IMAGE:
            data_from_dict = self.__blend_data.images
        elif type == Fpt_PackedLibrary_Type.TYPE_MODEL:
            #print("#DEBUG LoadFromEmbedded groups", [g for g in self.__blend_data.groups])
            data_from_dict = self.__blend_data.groups
        else:
            return None
        return data_from_dict.get(name)

    def LoadFromBlendLibrary(self, name, type, file):
        """
        name: 'fpmodel.fpl\\<object>.g'
        type: 'IMAGE', 'MODEL'
        file: '<addons>\\io_scene_fpx\\fpx_resource.blend'
        """
        #print("#DEBUG LoadFromBlendLibrary", name, type, file)

        with self.__blend_data.libraries.load(file) as (data_from, data_to):
            # imports data from library
            if type == Fpt_PackedLibrary_Type.TYPE_IMAGE:
                data_from_list = data_from.images
                name_temp = name

                if name_temp not in data_from.images:
                    return None
                # embed image data from library
                data_to.images = [name_temp, ]

            elif type == Fpt_PackedLibrary_Type.TYPE_MODEL:
                if name.endswith(".g"):
                    name_scene = "{}.s".format(name[:-2])
                    if name_scene not in data_from.scenes:
                        return None
                    # embed scene data from library
                    data_to.scenes = [name_scene, ]

                if name not in data_from.groups:
                    return None
                # embed group data from library
                data_to.groups = [name, ]

            else:
                return None

        # try to get internal data
        return self.LoadFromEmbedded(name, type)

    def LoadFromPathLibrary(self, name, type, lib, folder):
        """
        name: 'fpmodel.fpl\\<object>.g'
        type: 'IMAGE', 'MODEL'
        lib:  'fpmodel.fpl'
        folder: '.\\'
        """
        #print("#DEBUG LoadFromPathLibrary", name, type, lib, folder)

        filepath = path.join(folder, lib)
        if path.exists(filepath):
            FplImporter(
                    report=self.report,
                    verbose=self.verbose,
                    keep_temp=self.keep_temp,
                    use_library_filter=self.use_library_filter,
                    use_model_filter=self.use_model_filter,
                    use_model_adjustment=self.use_model_adjustment,
                    keep_name=False,
                    ).read(
                            self.__context,
                            filepath,
                            )

            return self.LoadFromEmbedded(name, type)

        return None

    def GetLinked(self, dictIn, dictOut, type, dst_sub_path_names):
        """
        loads external resources
        """
        if type not in self.use_library_filter:
            return

        for key, fpx_item in dictIn.items():
            if not fpx_item:
                continue

            fpx_item_name = fpx_item.get_value("name")
            fpx_item_name = FpxUtilities.toGoodName(fpx_item_name) ####

            if type == Fpt_PackedLibrary_Type.TYPE_IMAGE:
                blender_resource_name_format = FORMAT_IMAGE.format(FORMAT_RESOURCE)
            elif type == Fpt_PackedLibrary_Type.TYPE_MODEL:
                blender_resource_name_format = FORMAT_GROUP.format(FORMAT_RESOURCE)
            else:
                ## TODO
                continue

            lib_name = None

            if fpx_item.get_value("linked"):
                #print("#DEBUG GetLinked linked > ", type, fpx_item_name)

                linked_path = fpx_item.get_value("linked_path").lower()
                library_file, file_name = linked_path.split(Fpt_File_Reader.SEPARATOR)
                library_file = FpxUtilities.toGoodName(library_file)
                file_name = FpxUtilities.toGoodName(file_name)

                lib_name = library_file

                if file_name.endswith(".fpm") or file_name.endswith(".jpg") or file_name.endswith(".bmp") or file_name.endswith(".tga"):
                    file_name = file_name[:-4]

                if library_file == "":
                    prefix_library_file = PREFIX_LOCAL
                else:
                    prefix_library_file = library_file

                blender_resource_name = FpxUtilities.toGoodName(blender_resource_name_format.format(prefix_library_file, file_name))

                dictOut[fpx_item_name] = [ blender_resource_name, ]

            else:
                #print("#DEBUG GetLinked intern > ", type, fpx_item_name)

                blender_resource_name = FpxUtilities.toGoodName(blender_resource_name_format.format(PREFIX_EMBEDDED, fpx_item_name))
                dictOut[fpx_item_name] = [ blender_resource_name, ]
                item_data = dst_sub_path_names.get("data_{}".format(fpx_item_name))
                if item_data:
                    #print("#DEBUG", fpx_item_name, item_data[1]["sub_dir"])
                    active_scene = self.__context.screen.scene
                    FpmImporter(
                            report=self.report,
                            verbose=self.verbose,
                            keep_temp=self.keep_temp,
                            use_scene_per_model=True,
                            name_extra="",
                            use_model_filter=self.use_model_filter,
                            use_model_adjustment=self.use_model_adjustment,
                            keep_name=False,
                        ).read_ex(
                                blender_context=self.__context,
                                dst_sub_path_names=item_data[1],
                                model_name=FpxUtilities.toGoodName(FORMAT_RESOURCE.format(PREFIX_EMBEDDED, fpx_item_name)),
                                debug_data=[],
                                )
                    #rename_active_fpm(self.__context, blender_resource_name)
                    self.__context.screen.scene = active_scene

            self.LoadObject(blender_resource_name, type, lib_name)

    ###########################################################################

###############################################################################
def get_min_max(blender_object):
    min_x = max_x = min_y = max_y = min_z = max_z = None

    for vert in blender_object.data.vertices:
        x, y, z = vert.co

        if min_x is None:
            min_x = max_x = x
            min_y = max_y = y
            min_z = max_z = z
            continue

        if x < min_x:
            min_x = x
        elif x > max_x:
            max_x = x

        if y < min_y:
            min_y = y
        elif y > max_y:
            max_y = y

        if z < min_z:
            min_z = z
        elif z > max_z:
            max_z = z

    return min_x, min_y, min_z, max_x, max_y, max_z

def adjust_position(blender_context, blender_scene, fpx_model, fpx_model_type=None):
    if not blender_context.active_object:
        return
    fpx_type = fpx_model.get("type")
    has_mask = fpx_model.get("mask_model_data")

    blender_object = blender_context.active_object
    blender_parent = blender_object.parent

    # Fpm_Model_Type.OBJECT_OBJECT = 0
    ## Fpm_Model_Type.OBJECT_PEG = 1
    ## Fpm_Model_Type.OBJECT_ORNAMENT = 8

    # Fpm_Model_Type.CONTROL_BUMPER_BASE = 3
    # Fpm_Model_Type.CONTROL_BUMPER_CAP = 4

    ## Fpm_Model_Type.CONTROL_FLIPPER = 2
    # Fpm_Model_Type.CONTROL_PLUNGER = 7
    # Fpm_Model_Type.CONTROL_KICKER = 9
    # Fpm_Model_Type.CONTROL_DIVERTER = 18
    # Fpm_Model_Type.CONTROL_AUTOPLUNGER = 19
    # Fpm_Model_Type.CONTROL_POPUP = 20
    # Fpm_Model_Type.CONTROL_EMKICKER = 24

    # Fpm_Model_Type.TARGET_TARGET = 5
    # Fpm_Model_Type.TARGET_DROP = 6
    # Fpm_Model_Type.TARGET_VARI = 26

    # Fpm_Model_Type.TRIGGER_TRIGGER = 12
    # Fpm_Model_Type.TRIGGER_GATE = 15
    # Fpm_Model_Type.TRIGGER_SPINNER = 16
    # Fpm_Model_Type.TRIGGER_OPTO = 25

    ## Fpm_Model_Type.LIGHT_FLASHER = 13
    ## Fpm_Model_Type.LIGHT_BULB = 14

    ## Fpm_Model_Type.TOY_SPINNINGDISK = 23
    ## Fpm_Model_Type.TOY_CUSTOM = 17

    # Fpm_Model_Type.GUIDE_LANE = 10
    # Fpm_Model_Type.RUBBER_MODEL = 11
    # Fpm_Model_Type.RAMP_MODEL = 21
    # Fpm_Model_Type.RAMP_WIRE_CAP = 22

    # fpx_model_type=None
    # fpx_model_type="lo"
    # fpx_model_type="ma"
    # fpx_model_type="mi"

    # bumper ring = top + 31
    #
    #

    blender_location = None
    if fpx_type in {
            Fpm_Model_Type.OBJECT_PEG, #
            Fpm_Model_Type.OBJECT_ORNAMENT, # mask
            Fpm_Model_Type.CONTROL_BUMPER_BASE, #
            Fpm_Model_Type.CONTROL_FLIPPER, #
            Fpm_Model_Type.CONTROL_DIVERTER, #
            Fpm_Model_Type.CONTROL_AUTOPLUNGER,
            Fpm_Model_Type.CONTROL_KICKER, #
            Fpm_Model_Type.LIGHT_FLASHER, # mask
            Fpm_Model_Type.LIGHT_BULB, #
            Fpm_Model_Type.TRIGGER_OPTO, #
            Fpm_Model_Type.TOY_CUSTOM,
            }:
        static_offset = 0
        if has_mask:
            #align to top
            #blender_location = Vector((blender_parent.location.x, blender_parent.location.y, -blender_object.dimensions.z / 2.0))
            blender_location = Vector((blender_parent.location.x, blender_parent.location.y, get_min_max(blender_object)[2] + static_offset))
        else:
            # align to bottom
            #blender_location = Vector((blender_parent.location.x, blender_parent.location.y, blender_object.dimensions.z / 2.0))
            blender_location = Vector((blender_parent.location.x, blender_parent.location.y, get_min_max(blender_object)[5] + static_offset))

    elif fpx_type in {
            Fpm_Model_Type.CONTROL_BUMPER_CAP, #
            }:
        # align to bottom + static offset
        static_offset = 31.5
        #blender_location = Vector((blender_parent.location.x, blender_parent.location.y, (blender_object.dimensions.z / 2.0) + static_offset))
        blender_location = Vector((blender_parent.location.x, blender_parent.location.y, get_min_max(blender_object)[5] + static_offset))
    elif fpx_type in {
            Fpm_Model_Type.TOY_SPINNINGDISK, #
            }:
        # align to top + static offset
        static_offset = 0.25
        #blender_location = Vector((blender_parent.location.x, blender_parent.location.y, (-blender_object.dimensions.z / 2.0) + static_offset))
        blender_location = Vector((blender_parent.location.x, blender_parent.location.y, get_min_max(blender_object)[2] + static_offset))
    elif fpx_type in {
            Fpm_Model_Type.TRIGGER_TRIGGER, #
            }:
        # dont touch
        pass
    elif fpx_type in {
            Fpm_Model_Type.TRIGGER_SPINNER, #
            }:
        # static offset
        static_offset = 3
        blender_location = Vector((blender_parent.location.x, blender_parent.location.y, blender_parent.location.z + static_offset))
    elif fpx_type in {
            Fpm_Model_Type.TRIGGER_GATE, #
            }:
        # align to top + static offset
        static_offset = 6
        #blender_location = Vector((blender_parent.location.x, blender_parent.location.y, (-blender_object.dimensions.z / 2.0) + static_offset))
        blender_location = Vector((blender_parent.location.x, blender_parent.location.y, get_min_max(blender_object)[2] + static_offset))
    elif fpx_type in {
            Fpm_Model_Type.CONTROL_POPUP, #
            }:
        # align to top
        static_offset = 0
        #blender_location = Vector((blender_parent.location.x, blender_parent.location.y, (-blender_object.dimensions.z / 2.0) + static_offset))
        blender_location = Vector((blender_parent.location.x, blender_parent.location.y, get_min_max(blender_object)[2] + static_offset))
    elif fpx_type in {
            Fpm_Model_Type.TARGET_DROP, #
            }:
        # static offset
        static_offset = 12.5
        blender_location = Vector((blender_parent.location.x, blender_parent.location.y, blender_parent.location.z + static_offset))
    elif fpx_type in {
            Fpm_Model_Type.TARGET_TARGET, #
            }:
        # static offset
        static_offset = 15
        blender_location = Vector((blender_parent.location.x, blender_parent.location.y, blender_parent.location.z + static_offset))
    elif fpx_type in {
            Fpm_Model_Type.RAMP_WIRE_CAP, #
            }:
        # static offset
        static_offset = 13
        blender_location = Vector((blender_parent.location.x, blender_parent.location.y, blender_parent.location.z + static_offset))
    elif fpx_type in {
            Fpm_Model_Type.OBJECT_OBJECT, # wire ring
            }:
        # static offset
        static_offset = 11
        blender_location = Vector((blender_parent.location.x, blender_parent.location.y, blender_parent.location.z + static_offset))
    elif fpx_type in {
            Fpm_Model_Type.RUBBER_MODEL, #
            Fpm_Model_Type.GUIDE_LANE, #
            Fpm_Model_Type.RAMP_MODEL, #
            }:
        # align to bottom + static offset
        static_offset = 0
        blender_location = Vector((blender_parent.location.x, blender_parent.location.y, get_min_max(blender_object)[5] + static_offset))
    else:
        pass

    if blender_location:
        blender_scene.objects.active = blender_parent
        blender_scene.objects.active.location = blender_location
        blender_scene.update()

def remove_material(blender_context):
    if not blender_context.active_object:
        return
    active_object = blender_context.active_object
    blender_data = blender_context.blend_data

    used_materials = []
    used_textures = []
    used_images = []

    for material_slot in active_object.material_slots:
        if not material_slot or not material_slot.material:
            continue
        material = material_slot.material
        used_materials.append(material)
        for texture_slot in material.texture_slots:
            if not texture_slot or not texture_slot.texture:
                continue
            texture = texture_slot.texture
            used_textures.append(texture)
            if texture.type == 'IMAGE' and texture.image:
                used_images.append(texture.image)
                texture.image = None
            texture_slot.texture = None
        material_slot.material = None

    if ops.object.material_slot_remove.poll:
        ops.object.material_slot_remove()

    for material in used_materials:
        material.user_clear()
        blender_data.materials.remove(material)

    for texture in used_textures:
        texture.user_clear()
        blender_data.textures.remove(texture)

    for image in used_images:
        image.user_clear()
        blender_data.images.remove(image)

def rename_active_ms3d(blender_context, src_name, dst_name, dst_type=None):
    #print("#DEBUG rename_active_ms3d >", blender_context.active_object, src_name, dst_name, dst_type)
    if not blender_context.active_object:
        return

    data = blender_context.blend_data

    src_empty_object_name = FORMAT_EMPTY_OBJECT.format(src_name)
    src_mesh_object_name = FORMAT_MESH_OBJECT.format(src_name)
    src_mesh_name = FORMAT_MESH.format(src_name)
    src_armature_name = FORMAT_ARMATURE.format(src_name)
    src_armature_object_name = FORMAT_ARMATURE_OBJECT.format(src_name)
    src_action_name = FORMAT_ACTION.format(src_name)
    src_group_name = FORMAT_GROUP.format(src_name)

    if dst_type:
        dst_name = "{}.{}".format(dst_name, dst_type)

    dst_empty_object_name = FpxUtilities.toGoodName(FORMAT_EMPTY_OBJECT.format(dst_name))
    dst_mesh_object_name = FpxUtilities.toGoodName(FORMAT_MESH_OBJECT.format(dst_name))
    dst_mesh_name = FpxUtilities.toGoodName(FORMAT_MESH.format(dst_name))
    dst_armature_name = FpxUtilities.toGoodName(FORMAT_ARMATURE.format(dst_name))
    dst_armature_object_name = FpxUtilities.toGoodName(FORMAT_ARMATURE_OBJECT.format(dst_name))
    dst_action_name = FpxUtilities.toGoodName(FORMAT_ACTION.format(dst_name))
    dst_group_name = FpxUtilities.toGoodName(FORMAT_GROUP.format(dst_name))

    obj = blender_context.blend_data.objects.get(src_empty_object_name)
    if obj:
        obj.name = dst_empty_object_name

    obj = data.objects.get(src_mesh_object_name)
    if obj:
        obj.name = dst_mesh_object_name
        mod = obj.modifiers.get(src_armature_name)
        if mod:
            mod.name = dst_armature_name

    obj = data.objects.get(src_armature_object_name)
    if obj:
        obj.name = dst_armature_object_name

    obj = data.meshes.get(src_mesh_name)
    if obj:
        obj.name = dst_mesh_name

    obj = data.armatures.get(src_armature_name)
    if obj:
        obj.name = dst_armature_name

    obj = data.actions.get(src_action_name)
    if obj:
        obj.name = dst_action_name

    obj = data.groups.get(src_group_name)
    if obj:
        obj.name = dst_group_name

def rename_active_fpm(blender_context, dst_name):
    #print("#DEBUG rename_active_fpm >", blender_context.active_object, dst_name)
    if not blender_context.active_object:
        return
    blender_name = blender_context.active_object.name
    fpm_ext = "fpm"
    index = blender_name.find(".{}.".format(fpm_ext))
    if index < 0:
        index = blender_name.rfind(".")
        if index < 0:
            return

    src_name = blender_name[:index]

    #print("#DEBUG rename_active_fpm 2>", src_name, dst_name)

    #if src_name.endswith(".fpm"):
    #    src_name = src_name[:-4]

    src_scene_name = blender_context.scene.name

    blender_scene = blender_context.blend_data.scenes.get(src_scene_name)
    if blender_scene:
        blender_scene.name = FORMAT_SCENE.format(dst_name).lower()

    for object_type in {'', '.secondary', '.mask', '.reflection', '.collision', }:
        fpm_ext_ex = "{}{}".format(src_name, object_type)
        rename_active_ms3d(blender_context, fpm_ext_ex, dst_name, object_type);

def get_blend_resource_file_name():
    from importlib import ( find_loader, )
    from os import ( path, )

    module_path = find_loader('io_scene_fpx').path
    module_path, module_file = path.split(module_path)

    resource_blend = path.join(module_path, 'fpx_resource.blend')

    if not path.exists(resource_blend):
        print("#DEBUG", "resource not found!")

    return resource_blend



###############################################################################


###############################################################################
#234567890123456789012345678901234567890123456789012345678901234567890123456789
#--------1---------2---------3---------4---------5---------6---------7---------
# ##### END OF FILE #####
